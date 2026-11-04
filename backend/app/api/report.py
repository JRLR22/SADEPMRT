"""Informe PDF de un estudio."""

import json
import os
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.rate_limit import rate_limit_report
from app.database.base import get_db
from app.models.study import Study
from app.models.user import User
from app.services.records import _log_audit

router = APIRouter(tags=["Informes"])


@router.get("/studies/{study_id}/report")
async def generate_pdf_report(
    study_id:     str,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
    _rate:        None    = Depends(rate_limit_report),
):
    try:
        from sqlalchemy.orm import joinedload
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle,
            Paragraph, Spacer, Image as RLImage,
            KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.pdfgen import canvas as rl_canvas

        BLUE       = colors.HexColor('#1a56db')
        BLUE_LT    = colors.HexColor('#eff4ff')
        BLUE_MID   = colors.HexColor('#c7d7f9')
        RED        = colors.HexColor('#e02424')
        RED_LT     = colors.HexColor('#fef2f2')
        GREEN      = colors.HexColor('#0d9f6e')
        GREEN_LT   = colors.HexColor('#ecfdf5')
        GRAY_TEXT  = colors.HexColor('#4a5e7a')
        GRAY_DIM   = colors.HexColor('#8fa3bf')
        GRAY_BG    = colors.HexColor('#f8fafc')
        DARK       = colors.HexColor('#0f1f38')
        WHITE      = colors.white
        BORDER     = colors.HexColor('#e2e8f0')

        s = (
            db.query(Study)
            .options(joinedload(Study.predictions))
            .filter(Study.id == study_id)
            .first()
        )
        if not s:
            raise HTTPException(status_code=404, detail="Estudio no encontrado")
        if s.uploaded_by != current_user.id and current_user.role not in ['admin', 'radiologist']:
            raise HTTPException(status_code=403, detail="Sin permiso")
        if not s.predictions:
            raise HTTPException(status_code=400, detail="Sin predicciones")

        _log_audit(db, current_user.id, "report", "study", {'study_id': study_id})
        db.commit()

        pred     = s.predictions[0]
        probs    = json.loads(pred.probabilities)           if pred.probabilities           else {}
        clinical = json.loads(pred.clinical_interpretation) if pred.clinical_interpretation else {}

        is_normal     = 'normal' in pred.prediction.lower()
        result_color  = GREEN    if is_normal else RED
        result_bg     = GREEN_LT if is_normal else RED_LT
        result_label  = 'NORMAL — Sin hallazgos patológicos' if is_normal else f'POSITIVO — {pred.prediction.upper()}'

        disease_display = {
            'COVID':        'COVID-19',
            'PNEUMONIA':    'Neumonía',
            'TUBERCULOSIS': 'Tuberculosis',
        }.get(str(s.disease_type).upper().replace('DISEASETYPE.', ''), str(s.disease_type))

        # created_at está en utc; el reporte se imprime en hora de mazatlán
        from datetime import timezone
        import pytz
        mzt    = pytz.timezone('America/Mazatlan')
        fecha  = s.created_at.replace(tzinfo=timezone.utc).astimezone(mzt)
        fecha_str  = fecha.strftime('%d/%m/%Y %H:%M')
        gen_str    = datetime.now(mzt).strftime('%d/%m/%Y %H:%M:%S')

        def url_to_path(url: str):
            if not url:
                return None
            for marker in ['localhost:8000/uploads/', '127.0.0.1:8000/uploads/']:
                if marker in url:
                    relative = url.split(marker)[-1]
                    return os.path.join('uploads', relative.replace('/', os.sep))
            if url.startswith('uploads/'):
                return url
            return None

        class HeaderFooterCanvas(rl_canvas.Canvas):
            """Guarda las páginas antes de dibujarlas para poder poner "Página X de N"."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pages = []

            def showPage(self):
                self.pages.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                page_count = len(self.pages)
                for i, page in enumerate(self.pages):
                    self.__dict__.update(page)
                    self._draw_header_footer(i + 1, page_count)
                    super().showPage()
                super().save()

            def _draw_header_footer(self, page_num, page_count):
                w, h = letter

                self.setFillColor(BLUE)
                self.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)

                self.setFillColor(WHITE)
                self.setFont('Helvetica-Bold', 11)
                self.drawString(0.5 * inch, h - 0.35 * inch, 'rt26 — Sistema de Detección Pulmonar IA v2.0')

                self.setFont('Helvetica', 9)
                self.drawRightString(w - 0.5 * inch, h - 0.35 * inch, f'Página {page_num} de {page_count}')

                self.setStrokeColor(BLUE_MID)
                self.setLineWidth(0.5)
                self.line(0.5 * inch, h - 0.6 * inch, w - 0.5 * inch, h - 0.6 * inch)

                self.setStrokeColor(BORDER)
                self.setLineWidth(0.5)
                self.line(0.5 * inch, 0.55 * inch, w - 0.5 * inch, 0.55 * inch)

                self.setFillColor(GRAY_DIM)
                self.setFont('Helvetica', 8)
                self.drawString(0.5 * inch, 0.35 * inch,
                    f'Generado: {gen_str}  •  Este reporte es una herramienta de apoyo — no sustituye criterio médico')
                self.drawRightString(w - 0.5 * inch, 0.35 * inch,
                    f'Estudio ID: {str(s.id)[:8].upper()}')

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.6 * inch,
            leftMargin=0.6 * inch,
            topMargin=1.0 * inch,      # espacio para el header azul
            bottomMargin=0.85 * inch,  # espacio para el footer
        )

        styles = getSampleStyleSheet()

        def section_header(text):
            t = Table([[Paragraph(
                f'<font color="white"><b>{text}</b></font>',
                ParagraphStyle('sh', fontName='Helvetica-Bold', fontSize=10,
                               textColor=WHITE, leading=14)
            )]], colWidths=[6.8 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), BLUE),
                ('TOPPADDING',    (0,0),(-1,-1), 7),
                ('BOTTOMPADDING', (0,0),(-1,-1), 7),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('RIGHTPADDING',  (0,0),(-1,-1), 10),
                ('ROUNDEDCORNERS',(0,0),(-1,-1), [4, 4, 0, 0]),
            ]))
            return t

        def info_table(rows, col_widths=None):
            cw = col_widths or [1.8 * inch, 5.0 * inch]
            t  = Table(rows, colWidths=cw)
            style = [
                ('FONTNAME',      (0,0),(0,-1), 'Helvetica-Bold'),
                ('FONTNAME',      (1,0),(1,-1), 'Helvetica'),
                ('FONTSIZE',      (0,0),(-1,-1), 9),
                ('TEXTCOLOR',     (0,0),(0,-1), DARK),
                ('TEXTCOLOR',     (1,0),(1,-1), GRAY_TEXT),
                ('TOPPADDING',    (0,0),(-1,-1), 6),
                ('BOTTOMPADDING', (0,0),(-1,-1), 6),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('RIGHTPADDING',  (0,0),(-1,-1), 10),
                ('GRID',          (0,0),(-1,-1), 0.5, BORDER),
                ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ]
            t.setStyle(TableStyle(style))
            return t

        sp = lambda n: Spacer(1, n)

        elements = []

        elements.append(sp(0.15 * inch))
        elements.append(Paragraph(
            'REPORTE DE ANÁLISIS RADIOLÓGICO',
            ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18,
                           textColor=DARK, alignment=TA_CENTER, spaceAfter=4,
                           letterSpacing=0.5)
        ))
        elements.append(Paragraph(
            'Diagnóstico asistido por IA — Ensemble de 3 modelos de Deep Learning',
            ParagraphStyle('sub', fontName='Helvetica', fontSize=10,
                           textColor=GRAY_DIM, alignment=TA_CENTER)
        ))
        elements.append(sp(0.25 * inch))

        conf_pct = f'{pred.confidence * 100:.1f}%'
        risk_lvl = clinical.get('level', '—')
        risk_msg = clinical.get('recommendation', '—')

        result_table = Table(
            [[
                Paragraph(
                    f'<font size="11" color="{result_color.hexval() if hasattr(result_color,"hexval") else "#0d9f6e"}"><b>DIAGNÓSTICO</b></font><br/>'
                    f'<font size="16"><b>{result_label}</b></font>',
                    ParagraphStyle('res', fontName='Helvetica', fontSize=16,
                                   textColor=result_color, leading=22)
                ),
                Paragraph(
                    f'<font size="9" color="#8fa3bf">Confianza del modelo</font><br/>'
                    f'<font size="22"><b>{conf_pct}</b></font>',
                    ParagraphStyle('conf', fontName='Helvetica-Bold', fontSize=22,
                                   textColor=result_color, alignment=TA_CENTER, leading=28)
                ),
            ]],
            colWidths=[4.5 * inch, 2.3 * inch]
        )
        result_table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), result_bg),
            ('TOPPADDING',    (0,0),(-1,-1), 14),
            ('BOTTOMPADDING', (0,0),(-1,-1), 14),
            ('LEFTPADDING',   (0,0),(-1,-1), 14),
            ('RIGHTPADDING',  (0,0),(-1,-1), 14),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), [6, 6, 6, 6]),
            ('BOX',           (0,0),(-1,-1), 1.5, result_color),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ('LINEAFTER',     (0,0),(0,-1),  0.5, result_color),
        ]))
        elements.append(result_table)
        elements.append(sp(0.22 * inch))

        elements.append(section_header('INFORMACIÓN DEL ESTUDIO'))
        elements.append(info_table([
            ['ID del estudio:',  str(s.id)],
            ['Fecha del estudio:', fecha_str],
            ['Tipo de análisis:', disease_display],
            ['Paciente:',        s.patient_identifier or 'No especificado'],
            ['Modelo utilizado:', 'Ensemble (DenseNet121 + ResNet50 + EfficientNet-B0)'],
            ['Tiempo de proc.:', f'{pred.processing_time_ms:.0f} ms'],
        ]))
        elements.append(sp(0.22 * inch))

        img_items = [
            (s.original_image_url,        'Radiografía Original'),
            (pred.gradcam_image_url,       'Grad-CAM (activación)'),
            (pred.yolo_image_url,          'YOLO (detección)'),
            (pred.segmentation_image_url,  'U-Net (segmentación)'),
        ]

        img_size   = 1.4 * inch
        label_style = ParagraphStyle('lbl', fontName='Helvetica-Bold', fontSize=8,
                                     textColor=DARK, alignment=TA_CENTER)
        note_style  = ParagraphStyle('note', fontName='Helvetica', fontSize=7,
                                     textColor=GRAY_DIM, alignment=TA_CENTER)

        def make_img_cell(url, label, note=''):
            path = url_to_path(url) if url else None
            if path and os.path.exists(path):
                img = RLImage(path, width=img_size, height=img_size)
                img.hAlign = 'CENTER'
                return [img, Paragraph(label, label_style),
                        Paragraph(note, note_style) if note else Spacer(1, 1)]
            else:
                placeholder = Table(
                    [[Paragraph('N/D', ParagraphStyle('nd', fontName='Helvetica',
                                fontSize=9, textColor=GRAY_DIM, alignment=TA_CENTER))]],
                    colWidths=[img_size], rowHeights=[img_size]
                )
                placeholder.setStyle(TableStyle([
                    ('BACKGROUND', (0,0),(-1,-1), GRAY_BG),
                    ('BOX',        (0,0),(-1,-1), 0.5, BORDER),
                    ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
                ]))
                return [placeholder, Paragraph(label, label_style)]

        row1 = [make_img_cell(*img_items[0], 'Imagen sin procesar'),
                make_img_cell(*img_items[1], 'Rojo = mayor atencion IA')]
        row2 = [make_img_cell(*img_items[2], 'Bounding boxes'),
                make_img_cell(*img_items[3], 'Mascara de segmentacion')]

        def pair_to_table(left_cell, right_cell):
            col_w = (6.8 * inch) / 2
            t = Table(
                [[left_cell[0],  right_cell[0]],
                 [left_cell[1],  right_cell[1]],
                 [left_cell[2] if len(left_cell) > 2 else Spacer(1,1),
                  right_cell[2] if len(right_cell) > 2 else Spacer(1,1)]],
                colWidths=[col_w, col_w]
            )
            t.setStyle(TableStyle([
                ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
                ('VALIGN',        (0,0),(  -1,0), 'BOTTOM'),
                ('TOPPADDING',    (0,0),(-1,-1), 4),
                ('BOTTOMPADDING', (0,0),(-1,-1), 2),
            ]))
            return t

        elements.append(section_header('IMÁGENES DEL ANÁLISIS'))
        elements.append(KeepTogether([
            sp(6),
            pair_to_table(row1[0], row1[1]),
            sp(8),
            pair_to_table(row2[0], row2[1]),
            sp(0.22 * inch),
        ]))

        elements.append(section_header('PROBABILIDADES POR CLASE'))
        elements.append(sp(4))

        prob_rows = [
            [
                Paragraph(f'<b>{label}</b>',
                          ParagraphStyle('pl', fontName='Helvetica-Bold', fontSize=9, textColor=DARK)),
                Paragraph(f'{p*100:.2f}%',
                          ParagraphStyle('pv', fontName='Helvetica-Bold', fontSize=9,
                                         textColor=result_color if 'normal' not in label.lower() else GRAY_TEXT,
                                         alignment=TA_CENTER)),
            ]
            for label, p in probs.items()
        ]

        pt = Table(prob_rows, colWidths=[5.1 * inch, 1.7 * inch])
        pt.setStyle(TableStyle([
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('TOPPADDING',    (0,0),(-1,-1), 7),
            ('BOTTOMPADDING', (0,0),(-1,-1), 7),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
            ('RIGHTPADDING',  (0,0),(-1,-1), 12),
            ('GRID',          (0,0),(-1,-1), 0.5, BORDER),
            ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GRAY_BG]),
            ('ALIGN',         (1,0),(1,-1),  'CENTER'),
        ]))
        elements.append(pt)
        elements.append(sp(0.22 * inch))

        if clinical:
            lvl_color = {
                'Alto':  GREEN,
                'Medio': colors.HexColor('#d97706'),
                'Bajo':  RED,
            }.get(risk_lvl, GRAY_TEXT)

            clin_table = Table(
                [[
                    Paragraph(
                        f'<b>Nivel de confianza clínica: {risk_lvl}</b><br/>'
                        f'<font size="8" color="#4a5e7a">{clinical.get("message","")}</font>',
                        ParagraphStyle('cl', fontName='Helvetica', fontSize=10,
                                       textColor=lvl_color, leading=14)
                    ),
                    Paragraph(
                        f'<font size="8"><b>Recomendacion:</b></font><br/>'
                        f'<font size="8" color="#4a5e7a">{risk_msg}</font>',
                        ParagraphStyle('cr', fontName='Helvetica', fontSize=9,
                                       textColor=DARK, leading=13)
                    ),
                ]],
                colWidths=[3.4 * inch, 3.4 * inch]
            )
            clin_table.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(0,-1), BLUE_LT),
                ('BACKGROUND',    (1,0),(1,-1), GRAY_BG),
                ('TOPPADDING',    (0,0),(-1,-1), 10),
                ('BOTTOMPADDING', (0,0),(-1,-1), 10),
                ('LEFTPADDING',   (0,0),(-1,-1), 12),
                ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                ('BOX',           (0,0),(-1,-1), 0.5, BLUE_MID),
                ('LINEBEFORE',    (1,0),(1,-1),  0.5, BLUE_MID),
                ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ]))
            elements.append(section_header('INTERPRETACIÓN CLÍNICA'))
            elements.append(sp(4))
            elements.append(clin_table)
            elements.append(sp(0.22 * inch))

        aviso = Table(
            [[Paragraph(
                '<b>AVISO LEGAL:</b> Este sistema es una herramienta de apoyo diagnóstico basada en '
                'inteligencia artificial. Los resultados <b>NO</b> sustituyen el criterio clínico de un '
                'radiólogo o médico certificado. Todo estudio debe ser revisado por un profesional de la '
                'salud antes de tomar decisiones clínicas. El sistema no está certificado bajo NOM-024-SSA3.',
                ParagraphStyle('av', fontName='Helvetica', fontSize=8,
                               textColor=colors.HexColor('#92400e'), leading=12,
                               alignment=TA_JUSTIFY)
            )]],
            colWidths=[6.8 * inch]
        )
        aviso.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#fffbeb')),
            ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor('#d97706')),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
            ('RIGHTPADDING',  (0,0),(-1,-1), 12),
            ('TOPPADDING',    (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), [4, 4, 4, 4]),
        ]))
        elements.append(aviso)

        doc.build(elements, canvasmaker=HeaderFooterCanvas)

        buffer.seek(0)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return StreamingResponse(
            buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename=reporte_{s.disease_type}_{ts}.pdf'}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'PDF error: {e}')
        raise HTTPException(status_code=500, detail=str(e))
