import React, { useState, useRef, useEffect } from 'react';
import { Bot, X, Send, RotateCcw, Paperclip } from 'lucide-react';


/**
 * ChatWidget - Asistente IA flotante para rt26
 * Se conecta a agent_api.py (puerto 8001) vía POST /chat
 *
 * Uso en App.js:
 *   import ChatWidget from './ChatWidget';
 *   ...
 *   <ChatWidget />   (dentro del layout principal, antes del último </div>)
 */

const AGENT_API_URL = 'http://localhost:8001';
const AZUL = '#1a56db';

// ---------- Renderizador de Markdown simple (sin dependencias externas) ----------
// Soporta: ### encabezados, **negritas**, - listas. Suficiente para los
// reportes que genera el agente, sin necesitar instalar una librería nueva.

function renderInline(text, keyPrefix) {
  const partes = text.split(/(\*\*[^*]+\*\*)/g);
  return partes.map((parte, i) => {
    if (parte.startsWith('**') && parte.endsWith('**')) {
      return <strong key={`${keyPrefix}-${i}`}>{parte.slice(2, -2)}</strong>;
    }
    return <span key={`${keyPrefix}-${i}`}>{parte}</span>;
  });
}

function renderMarkdown(texto) {
  const lineas = texto.split('\n');
  return lineas.map((linea, i) => {
    if (linea.startsWith('### ')) {
      return (
        <div key={i} style={{ fontWeight: 700, fontSize: 13, marginTop: 8, marginBottom: 3 }}>
          {renderInline(linea.slice(4), i)}
        </div>
      );
    }
    if (linea.trim().startsWith('- ')) {
      return (
        <div key={i} style={{ marginLeft: 10, display: 'flex', gap: 4 }}>
          <span>•</span>
          <span>{renderInline(linea.trim().slice(2), i)}</span>
        </div>
      );
    }
    if (linea.trim() === '') {
      return <div key={i} style={{ height: 6 }} />;
    }
    return <div key={i}>{renderInline(linea, i)}</div>;
  });
}

function nivelColor(top) {
  const label = (top.label || '').toLowerCase();
  if (label.includes('normal') || label.includes('negativo')) {
    return { fondo: '#ecfdf5', borde: '#10b981', texto: '#065f46' };
  }
  if (top.confidence >= 0.85) {
    return { fondo: '#fef2f2', borde: '#ef4444', texto: '#991b1b' };
  }
  return { fondo: '#fffbeb', borde: '#f59e0b', texto: '#92400e' };
}

function ReporteCard({ reporte }) {
  const top = reporte.ranking[0];
  const otras = reporte.ranking.slice(1);
  const colores = nivelColor(top);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        backgroundColor: colores.fondo,
        border: `1px solid ${colores.borde}`,
        borderRadius: 8,
        padding: '10px 12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <div style={{ fontSize: 10, color: colores.texto, fontWeight: 700, textTransform: 'uppercase' }}>
            Diagnóstico principal
          </div>
          <div style={{ fontSize: 15, fontWeight: 700, color: colores.texto }}>
            {top.disease} — {top.label}
          </div>
        </div>
        <div style={{ fontSize: 20, fontWeight: 800, color: colores.texto }}>
          {(top.confidence * 100).toFixed(1)}%
        </div>
      </div>

      {otras.length > 0 && (
        <div style={{ fontSize: 11, color: '#475569' }}>
          {otras.map((r, i) => (
            <span key={i} style={{ marginRight: 10 }}>
              {r.disease}: {(r.confidence * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {reporte.gradcam_url && (
          <a href={reporte.gradcam_url} target="_blank" rel="noreferrer" style={{ textAlign: 'center' }}>
            <img src={reporte.gradcam_url} alt="Grad-CAM" style={styles.imgReporte} />
            <div style={styles.imgLabel}>Grad-CAM</div>
          </a>
        )}
        {reporte.yolo_url && (
          <a href={reporte.yolo_url} target="_blank" rel="noreferrer" style={{ textAlign: 'center' }}>
            <img src={reporte.yolo_url} alt="YOLO" style={styles.imgReporte} />
            <div style={styles.imgLabel}>YOLO ({reporte.yolo_detections.length})</div>
          </a>
        )}
        {reporte.segmentation_url && (
          <a href={reporte.segmentation_url} target="_blank" rel="noreferrer" style={{ textAlign: 'center' }}>
            <img src={reporte.segmentation_url} alt="Segmentación" style={styles.imgReporte} />
            <div style={styles.imgLabel}>U-Net</div>
          </a>
        )}
      </div>

      <details style={{ fontSize: 11, color: '#334155' }}>
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Ver detalle técnico del análisis</summary>
        <div style={{ marginTop: 6 }}>
          <div style={{ fontWeight: 600, marginBottom: 2 }}>Votación del ensemble ({top.disease}):</div>
          {Object.entries(reporte.ensemble_detail || {}).map(([modelo, probs]) => (
            Array.isArray(probs) && probs.length > 1
              ? <div key={modelo}>• {modelo}: {(probs[1] * 100).toFixed(1)}%</div>
              : null
          ))}
          <div style={{ fontWeight: 600, margin: '6px 0 2px' }}>
            Hallazgos YOLO: {reporte.yolo_detections.length}
          </div>
          {reporte.yolo_detections.map((d, i) => (
            <div key={i}>• {d.class_name} ({(d.confidence * 100).toFixed(1)}%)</div>
          ))}
          <div style={{ marginTop: 6, fontStyle: 'italic', color: '#64748b' }}>
            Estudio ID: {reporte.study_id}
          </div>
        </div>
      </details>
    </div>
  );
}

export default function ChatWidget() {
  const [abierto, setAbierto] = useState(false);
  const [mensajes, setMensajes] = useState([
    { role: 'assistant', content: '¡Hola! Puedo explicarte diagnósticos, buscar información o resolver dudas sobre el sistema. ¿En qué te ayudo?' }
  ]);
const [input, setInput] = useState('');
const [cargando, setCargando] = useState(false);
const [imagen, setImagen] = useState(null);
const scrollRef = useRef(null);
const fileInputRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [mensajes, cargando]);

  const enviarMensaje = async () => {
    const pregunta = input.trim();
    if (!pregunta || cargando) return;

    setMensajes(prev => [...prev, { role: 'user', content: pregunta }]);
    setInput('');
    setCargando(true);

    try {
      const resp = await fetch(`${AGENT_API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pregunta }),
      });

      if (!resp.ok) {
        throw new Error(`Error del servidor: ${resp.status}`);
      }

      const data = await resp.json();
      setMensajes(prev => [...prev, { role: 'assistant', content: data.respuesta }]);
    } catch (err) {
      setMensajes(prev => [...prev, {
        role: 'assistant',
        content: 'No pude conectarme con el asistente en este momento. Inténtalo de nuevo en unos instantes.'
      }]);
    } finally {
      setCargando(false);
    }
  };

  const resetearChat = async () => {
  if (cargando) return;
  try {
    await fetch(`${AGENT_API_URL}/reset`, { method: 'POST' });
  } catch (err) {
    // aunque falle el backend, igual limpiamos la vista local
  }
  setMensajes([
    { role: 'assistant', content: '¡Hola! Puedo explicarte diagnósticos, buscar información o resolver dudas sobre el sistema. ¿En qué te ayudo?' }
  ]);
};

const enviarImagen = async () => {
  if (!imagen || cargando) return;

  setMensajes(prev => [...prev, { role: 'user', content: `📎 ${imagen.name}` }]);
  setCargando(true);

  const formData = new FormData();
  formData.append('file', imagen);
  formData.append('pregunta', input.trim());

  try {
    const resp = await fetch(`${AGENT_API_URL}/chat-imagen`, {
      method: 'POST',
      body: formData,
    });
    if (!resp.ok) throw new Error(`Error del servidor: ${resp.status}`);
    const data = await resp.json();
    setMensajes(prev => [...prev, { role: 'assistant', content: data.respuesta, reporte: data.reporte }]);
  } catch (err) {
    setMensajes(prev => [...prev, {
      role: 'assistant',
      content: 'No pude analizar la imagen en este momento. Inténtalo de nuevo en unos instantes.'
    }]);
  } finally {
    setCargando(false);
    setImagen(null);
    setInput('');
  }
};


  const manejarEnter = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      enviarMensaje();
    }
  };

  return (
    <div style={styles.contenedor}>
      {abierto && (
        <div className="chat-panel" style={styles.panel} role="dialog" aria-label="Asistente IA">
          {/* Header */}
          <div style={styles.header}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Bot size={18} />
              <span style={{ fontWeight: 600, fontSize: 14 }}>Asistente IA — rt26</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                onClick={resetearChat}
                style={styles.botonCerrar}
                aria-label="Nueva conversación"
                title="Nueva conversación"
              >
                <RotateCcw size={15} />
              </button>
              <button
                onClick={() => setAbierto(false)}
                style={styles.botonCerrar}
                aria-label="Cerrar chat"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Mensajes */}
          <div ref={scrollRef} style={styles.mensajesArea}>
            {mensajes.map((m, i) => (
              <div
                key={i}
                style={{
                  ...styles.burbuja,
                  ...(m.role === 'user' ? styles.burbujaUser : styles.burbujaAssistant),
                  ...(m.reporte ? { maxWidth: '95%' } : {}),
                }}
              >
                {m.reporte
                  ? <ReporteCard reporte={m.reporte} />
                  : (m.role === 'assistant' ? renderMarkdown(m.content) : m.content)}
              </div>
            ))}
            {cargando && (
              <div style={{ ...styles.burbuja, ...styles.burbujaAssistant, fontStyle: 'italic' }}>
                Pensando...
              </div>
            )}
          </div>

          {/* Input */}
<div style={styles.inputArea}>
  <input
    type="file"
    accept="image/*,.dcm"
    ref={fileInputRef}
    style={{ display: 'none' }}
    onChange={(e) => setImagen(e.target.files[0] || null)}
  />
  <button
    onClick={() => fileInputRef.current.click()}
    style={styles.botonAdjuntar}
    disabled={cargando}
    aria-label="Adjuntar radiografía"
    title="Adjuntar radiografía"
  >
    <Paperclip size={15} />
  </button>
  <textarea
    value={input}
    onChange={(e) => setInput(e.target.value)}
    onKeyDown={manejarEnter}
    placeholder={imagen ? `Imagen: ${imagen.name} (opcional: agrega una pregunta)` : "Escribe tu pregunta..."}
    rows={1}
    style={styles.textarea}
    disabled={cargando}
  />
  <button
    onClick={imagen ? enviarImagen : enviarMensaje}
    disabled={cargando || (!imagen && !input.trim())}
    aria-label="Enviar mensaje"
    style={{
      ...styles.botonEnviar,
      opacity: cargando || (!imagen && !input.trim()) ? 0.5 : 1,
    }}
  >
    <Send size={14} />
  </button>
</div>
        </div>
      )}

      {/* Botón flotante */}
      <button
        onClick={() => setAbierto(!abierto)}
        style={styles.botonFlotante}
        aria-label={abierto ? 'Cerrar chat' : 'Abrir chat'}
        aria-expanded={abierto}
      >
        {abierto ? <X size={22} /> : <Bot size={24} />}
      </button>
    </div>
  );
}

const styles = {
  contenedor: {
    position: 'fixed',
    bottom: '24px',
    right: '24px',
    zIndex: 9999,
    fontFamily: "'DM Sans', Helvetica, Arial, sans-serif",
  },
  botonFlotante: {
    width: '56px',
    height: '56px',
    borderRadius: '50%',
    backgroundColor: AZUL,
    color: 'white',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  panel: {
    position: 'absolute',
    bottom: '70px',
    right: '0',
    height: '480px',
    backgroundColor: 'white',
    borderRadius: '12px',
    boxShadow: '0 8px 30px rgba(0,0,0,0.3)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    border: '1px solid #e2e8f0',
  },
  header: {
    backgroundColor: AZUL,
    color: 'white',
    padding: '12px 16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  botonCerrar: {
    background: 'none',
    border: 'none',
    color: 'white',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
  },
  mensajesArea: {
    flex: 1,
    overflowY: 'auto',
    padding: '12px',
    backgroundColor: '#f8fafc',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  burbuja: {
    padding: '8px 12px',
    borderRadius: '10px',
    fontSize: '13px',
    lineHeight: 1.4,
    maxWidth: '85%',
    wordBreak: 'break-word',
  },
  burbujaUser: {
    backgroundColor: AZUL,
    color: 'white',
    alignSelf: 'flex-end',
    whiteSpace: 'pre-wrap',
  },
  burbujaAssistant: {
    backgroundColor: 'white',
    color: '#0f1f38',
    border: '1px solid #e2e8f0',
    alignSelf: 'flex-start',
  },
  inputArea: {
    display: 'flex',
    gap: '8px',
    padding: '10px',
    borderTop: '1px solid #e2e8f0',
    backgroundColor: 'white',
  },
  textarea: {
    flex: 1,
    resize: 'none',
    border: '1px solid #e2e8f0',
    borderRadius: '8px',
    padding: '8px',
    fontSize: '13px',
    fontFamily: 'inherit',
    outline: 'none',
  },
  botonEnviar: {
    backgroundColor: AZUL,
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    padding: '0 14px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  botonAdjuntar: {
  backgroundColor: 'white',
  color: AZUL,
  border: '1px solid #e2e8f0',
  borderRadius: '8px',
  padding: '0 10px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
},
imgReporte: {
  width: 88,
  height: 88,
  objectFit: 'cover',
  borderRadius: 6,
  border: '1px solid #e2e8f0',
  cursor: 'pointer',
},
imgLabel: {
  fontSize: 10,
  color: '#64748b',
  marginTop: 2,
},
};