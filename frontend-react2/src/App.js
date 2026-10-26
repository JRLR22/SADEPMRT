import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Tooltip, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid,
  AreaChart, Area,
} from 'recharts';
import {
  Activity, LayoutDashboard, ClipboardList, Users, Stethoscope,
  LogOut, Upload, Search, CheckCircle, AlertTriangle, Clock,
  Shield, Eye, ChevronRight,
  Hospital, Microscope, Wind, X, Check, HelpCircle,
  Download, Share2,
  RefreshCw, AlertCircle, Cpu, Layers
} from 'lucide-react';
import './App.css';
import './ui.css';
import ChatWidget from './ChatWidget';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const TZ      = { timeZone: 'America/Mazatlan' };
const getToken    = () => localStorage.getItem('token');
const authFetch = (url, opts = {}) =>
  fetch(url, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${getToken()}` } })
    .then(res => {
      if (res.status === 401) {
        localStorage.removeItem('token');
        try { sessionStorage.setItem('session_expired', '1'); } catch { /* sin storage: se omite el aviso */ }
        window.location.href = '/';
      }
      return res;
    });
const extractStudies = (data) => {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.studies)) return data.studies;
  return [];
};
// Cuenta de demostración opcional (definir en .env; ver README del frontend)
const DEMO_CREDS = process.env.REACT_APP_DEMO_EMAIL && process.env.REACT_APP_DEMO_PASSWORD
  ? { email: process.env.REACT_APP_DEMO_EMAIL, password: process.env.REACT_APP_DEMO_PASSWORD }
  : null;
const fmtDate  = (iso) => new Date(iso).toLocaleString('es-MX', TZ);
const fmtShort = (iso) => new Date(iso).toLocaleDateString('es-MX', TZ);

// Pasa a texto legible lo que devuelve el backend ("PNEUMONIA", "0 hallazgo(s)"…)
const DISEASE_LABELS = { PNEUMONIA: 'Neumonía', COVID: 'COVID-19', TUBERCULOSIS: 'Tuberculosis' };
const prettyText = (text = '') => text
  .replace(/YOLO detectó 0 hallazgo\(s\)/g, 'YOLO no detectó lesiones focales')
  .replace(/YOLO detectó (\d+) hallazgo\(s\)/g, (_, n) => `YOLO detectó ${n} ${n === '1' ? 'hallazgo' : 'hallazgos'}`)
  .replace(/\bPNEUMONIA\b/g, DISEASE_LABELS.PNEUMONIA)
  .replace(/\bCOVID\b(?!-)/g, DISEASE_LABELS.COVID)
  .replace(/\bTUBERCULOSIS\b/g, DISEASE_LABELS.TUBERCULOSIS);

const takeSessionExpiredNotice = () => {
  try {
    if (sessionStorage.getItem('session_expired')) {
      sessionStorage.removeItem('session_expired');
      return 'Tu sesión ha expirado. Inicia sesión nuevamente.';
    }
  } catch { /* sin storage */ }
  return null;
};

function ErrorState({ message, onRetry }) {
  return (
    <div role="alert" style={{
      padding: '24px 20px', borderRadius: T.radius, background: T.dangerLt, color: T.danger,
      border: '1px solid #fecaca', display: 'flex', alignItems: 'center', gap: 12,
      flexWrap: 'wrap', fontFamily: T.font, fontSize: 13,
    }}>
      <AlertCircle size={16} />
      <span style={{ flex: 1, minWidth: 180 }}>{message}</span>
      {onRetry && (
        <button onClick={onRetry} style={{
          display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 8,
          cursor: 'pointer', fontSize: 12, fontWeight: 600, fontFamily: T.font,
          background: T.surface, color: T.danger, border: '1px solid #fecaca',
        }}>
          <RefreshCw size={12} /> Reintentar
        </button>
      )}
    </div>
  );
}

/* ─── CSS-in-JS tokens ──────────────────────────────────────────────────── */
const T = {
  bg:        '#f0f4f9',
  surface:   '#ffffff',
  border:    '#e2e8f0',
  borderDim: '#edf0f4',
  text:      '#0f1f38',
  textMid:   '#4a5e7a',
  textDim:   '#8fa3bf',
  primary:   '#1a56db',
  primaryDk: '#1340b0',
  primaryLt: '#eff4ff',
  success:   '#0d9f6e',
  successLt: '#ecfdf5',
  danger:    '#e02424',
  dangerLt:  '#fef2f2',
  warn:      '#d97706',
  warnLt:    '#fffbeb',
  font:      "'DM Sans', 'Segoe UI', sans-serif",
  mono:      "'DM Mono', monospace",
  radius:    10,
  radiusLg:  16,
  shadow:    '0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05)',
  shadowMd:  '0 4px 24px rgba(0,0,0,0.10)',
};

/* ─── Shared style helpers ───────────────────────────────────────────────── */
const card = (extra = {}) => ({
  background: T.surface,
  border: `1px solid ${T.border}`,
  borderRadius: T.radius,
  padding: 20,
  marginBottom: 14,
  boxShadow: T.shadow,
  ...extra,
});

const badge = (color, bg, extra = {}) => ({
  display: 'inline-flex', alignItems: 'center', gap: 4,
  fontSize: 11, fontWeight: 700, padding: '3px 10px',
  borderRadius: 20, color, background: bg,
  letterSpacing: '0.02em', ...extra,
});

const btn = (variant = 'primary', extra = {}) => {
  const variants = {
    primary: { background: T.primary, color: '#fff', border: 'none' },
    ghost:   { background: 'transparent', color: T.textMid, border: `1px solid ${T.border}` },
    danger:  { background: T.dangerLt, color: T.danger, border: `1px solid #fecaca` },
  };
  return {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    padding: '8px 16px', borderRadius: 8, cursor: 'pointer',
    fontSize: 13, fontWeight: 600,
    fontFamily: T.font, ...variants[variant], ...extra,
  };
};

/* ─── FHIR export ────────────────────────────────────────────────────────── */
function exportFHIR(study, pred) {
  const isPos = pred && !pred.prediction?.toLowerCase().includes('normal');
  const fhir  = {
    resourceType: "DiagnosticReport",
    id: study.id,
    status: pred?.is_validated ? "final" : "preliminary",
    code: { text: "Radiografía de tórax — Análisis IA" },
    subject: { display: study.patient_identifier || "Paciente sin ID" },
    effectiveDateTime: study.created_at,
    conclusion: isPos
      ? `Hallazgo positivo: ${pred?.prediction}. Confianza IA: ${(pred?.confidence * 100).toFixed(1)}%.`
      : `Sin hallazgos patológicos. Confianza IA: ${(pred?.confidence * 100).toFixed(1)}%.`,
    extension: [{ url: "http://sistemapulmonar.mx/fhir/extension",
      valueString: JSON.stringify({ models: ["DenseNet121","ResNet50","EfficientNet-B0"],
        confidence: pred?.confidence, validation_result: pred?.validation_result }) }]
  };
  const blob = new Blob([JSON.stringify(fhir, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `FHIR_${study.id.slice(0,8)}.json`; a.click();
  URL.revokeObjectURL(url);
}

/* ════════════════════════════════════════════════════════════════════════════
   LOGIN
══════════════════════════════════════════════════════════════════════════════*/
function Login({ onLogin }) {
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState(takeSessionExpiredNotice);
  const [loading, setLoading]   = useState(false);

  const handleLogin = async (creds = { email, password }) => {
    setLoading(true); setError(null);
    try {
      const res  = await fetch(`${API_URL}/auth/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(creds),
      });
      if (!res.ok) throw new Error('Credenciales incorrectas');
      const data = await res.json();
      localStorage.setItem('token', data.access_token);
      onLogin(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="login-screen" style={{
      minHeight: '100vh', display: 'flex', fontFamily: T.font,
      background: 'linear-gradient(135deg, #0d1f3c 0%, #1a3a6b 50%, #0e2d5e 100%)',
    }}>
      {/* Left panel */}
      <div className="login-left" style={{
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', color: '#fff',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 48 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 12,
            background: 'rgba(255,255,255,0.15)', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={24} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.01em' }}>rt26</div>
            <div style={{ fontSize: 11, opacity: 0.6, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Sistema de Diagnóstico</div>
          </div>
        </div>
        <h1 className="login-title" style={{ fontSize: 42, fontWeight: 800, lineHeight: 1.15, margin: '0 0 20px', letterSpacing: '-0.02em' }}>
          Diagnóstico pulmonar<br />asistido por IA
        </h1>
        <p style={{ fontSize: 16, opacity: 0.7, lineHeight: 1.7, maxWidth: 420, margin: 0 }}>
          Ensemble de tres modelos de deep learning para detección de neumonía,
          COVID-19 y tuberculosis con Grad-CAM, YOLO y segmentación U-Net.
        </p>
        <div style={{ display: 'flex', gap: '16px 32px', marginTop: 48, flexWrap: 'wrap' }}>
          {[['97.6%', 'Precisión promedio'], ['< 3s', 'Tiempo de análisis'], ['3', 'Enfermedades detectadas']].map(([v, l]) => (
            <div key={l}>
              <div style={{ fontSize: 24, fontWeight: 800 }}>{v}</div>
              <div style={{ fontSize: 12, opacity: 0.55, marginTop: 2 }}>{l}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel */}
      <div className="login-right" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(255,255,255,0.04)', backdropFilter: 'blur(20px)',
        borderLeft: '1px solid rgba(255,255,255,0.08)',
      }}>
        <div className="login-form">
          <h2 style={{ color: '#fff', fontSize: 22, fontWeight: 700, margin: '0 0 8px' }}>Iniciar sesión</h2>
          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: 13, margin: '0 0 32px' }}>
            Ingresa tus credenciales para continuar
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input
              type="email" placeholder="Correo institucional"
              value={email} onChange={e => setEmail(e.target.value)}
              style={{
                padding: '12px 16px', borderRadius: 10, fontSize: 14,
                background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
                color: '#fff', fontFamily: T.font, outline: 'none',
              }}
            />
            <input
              type="password" placeholder="Contraseña"
              value={password} onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleLogin()}
              style={{
                padding: '12px 16px', borderRadius: 10, fontSize: 14,
                background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
                color: '#fff', fontFamily: T.font, outline: 'none',
              }}
            />
          </div>
          {error && (
            <div style={{
              marginTop: 12, padding: '10px 14px', borderRadius: 8,
              background: 'rgba(224,36,36,0.15)', color: '#fca5a5',
              fontSize: 13, display: 'flex', alignItems: 'center', gap: 8,
            }} role="alert">
              <AlertCircle size={14} /> {error}
            </div>
          )}
          <button
            onClick={() => handleLogin()} disabled={loading}
            style={{
              marginTop: 20, width: '100%', padding: '13px',
              borderRadius: 10, border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
              background: loading ? 'rgba(26,86,219,0.5)' : T.primary,
              color: '#fff', fontSize: 14, fontWeight: 700, fontFamily: T.font,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}>
            {loading ? <><RefreshCw size={14} style={{ animation: 'spin 1s linear infinite' }} /> Ingresando...</> : <>Ingresar <ChevronRight size={16} /></>}
          </button>
          {DEMO_CREDS && (
            <button
              onClick={() => handleLogin(DEMO_CREDS)} disabled={loading}
              style={{
                marginTop: 10, width: '100%', padding: '13px',
                borderRadius: 10, cursor: loading ? 'not-allowed' : 'pointer',
                background: 'transparent', border: '1px solid rgba(255,255,255,0.25)',
                color: '#fff', fontSize: 14, fontWeight: 600, fontFamily: T.font,
              }}>
              Probar demo sin cuenta
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   ROLE BADGE
══════════════════════════════════════════════════════════════════════════════*/
const ROLE_META = {
  admin:       { label: 'Admin',      color: '#6d28d9', bg: '#ede9fe' },
  radiologist: { label: 'Radiólogo',  color: T.primary, bg: T.primaryLt },
  user:        { label: 'Médico',     color: T.success, bg: T.successLt },
};
function RoleBadge({ role }) {
  const m = ROLE_META[role];
  if (!m) return null;
  return <span style={badge(m.color, m.bg)}><Shield size={10} />{m.label}</span>;
}

/* ════════════════════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════════════════════*/
function Sidebar({ active, onChange, onLogout, user, pendingCount = 0 }) {
  const canValidate = user?.role === 'admin' || user?.role === 'radiologist';
  const items = [
    { key: 'analysis',    Icon: Microscope,     label: 'Análisis'   },
    { key: 'dashboard',   Icon: LayoutDashboard, label: 'Dashboard'  },
    { key: 'history',     Icon: ClipboardList,   label: 'Historial'  },
    { key: 'patients',    Icon: Users,           label: 'Pacientes'  },
    ...(canValidate ? [{ key: 'radiologist', Icon: Stethoscope, label: 'Radiólogo', badge: pendingCount }] : []),
  ];

  return (
    <div className="sidebar" style={{
      background: T.text, color: '#fff',
      display: 'flex', flexDirection: 'column',
      position: 'sticky', top: 0, height: '100vh', flexShrink: 0,
      fontFamily: T.font,
    }}>
      {/* Logo */}
      <div className="sb-head" style={{ padding: '24px 20px 20px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 9, flexShrink: 0,
            background: T.primary, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={18} color="#fff" />
          </div>
          <div className="sb-hide">
            <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: '-0.01em' }}>rt26</div>
            <div style={{ fontSize: 10, opacity: 0.45, letterSpacing: '0.04em', textTransform: 'uppercase' }}>v2.0</div>
          </div>
        </div>
        {user && (
          <div className="sb-hide" style={{
            marginTop: 16, padding: '10px 12px', borderRadius: 8,
            background: 'rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', gap: 4,
          }}>
            <div style={{ fontSize: 12, opacity: 0.6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {user.email}
            </div>
            <RoleBadge role={user.role} />
          </div>
        )}
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '12px 10px' }}>
        {items.map(({ key, Icon, label, badge: badgeCount }) => {
          const isActive = active === key;
          return (
            <button key={key} onClick={() => onChange(key)}
              className={`nav-btn${isActive ? ' active' : ''}`}
              aria-current={isActive ? 'page' : undefined}
              aria-label={label} title={label}
              style={{
                display: 'flex', alignItems: 'center', gap: 10, width: '100%',
                padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
                fontFamily: T.font, fontSize: 13, marginBottom: 2,
              }}>
              <Icon size={16} style={{ flexShrink: 0 }} />
              <span className="sb-hide" style={{ flex: 1, textAlign: 'left' }}>{label}</span>
              {badgeCount > 0 && (
                <span className="nav-badge" style={{
                  background: T.danger, color: '#fff', borderRadius: 20,
                  fontSize: 10, fontWeight: 700, padding: '1px 7px',
                }}>{badgeCount}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Logout */}
      <div style={{ padding: '12px 10px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
        <button onClick={onLogout} className="logout-btn" aria-label="Cerrar sesión" title="Cerrar sesión" style={{
          display: 'flex', alignItems: 'center', gap: 10, width: '100%',
          padding: '10px 12px', borderRadius: 8, border: 'none', cursor: 'pointer',
          fontFamily: T.font, fontSize: 13, color: 'rgba(255,255,255,0.4)',
          background: 'transparent',
        }}>
          <LogOut size={16} style={{ flexShrink: 0 }} /><span className="sb-hide">Cerrar sesión</span>
        </button>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   KPI CARD
══════════════════════════════════════════════════════════════════════════════*/
function KpiCard({ Icon, label, value, color, subtitle }) {
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: T.radius,
      padding: 20,
      marginBottom: 14,
      boxShadow: T.shadow,
      flex: 1,
      minWidth: 130,
      borderTop: `3px solid ${color}`,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: color + '18',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <Icon size={16} color={color} />
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: T.text, letterSpacing: '-0.02em' }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: T.textMid, fontWeight: 500 }}>{label}</div>
      {subtitle && (
        <div style={{ fontSize: 11, color: T.textDim }}>{subtitle}</div>
      )}
    </div>
  );
}


const COLORS = [T.primary, T.danger, T.warn];
const NAMES  = ['Neumonía', 'COVID-19', 'Tuberculosis'];

function DonutChart({ data }) {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  const total  = data.reduce((s, d) => s + (d.total || 0), 0);

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }

    const values  = data.map(d => d.total || 0);
    const allZero = values.every(v => v === 0);

    chartRef.current = new window.Chart(canvasRef.current, {
      type: 'doughnut',
      data: {
        labels: NAMES,
        datasets: [{
          data: allZero ? [1, 1, 1] : values,
          backgroundColor: allZero ? ['#e2e8f0', '#e2e8f0', '#e2e8f0'] : COLORS,
          borderColor: '#ffffff',
          borderWidth: 3,
          hoverOffset: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: !allZero,
            callbacks: {
              label: (ctx) => {
                const val = ctx.parsed;
                const pct = total > 0 ? ((val / total) * 100).toFixed(1) : '0.0';
                return `  ${ctx.label}: ${val} (${pct}%)`;
              },
            },
            backgroundColor: 'rgba(15,31,56,0.92)',
            padding: 10,
            cornerRadius: 8,
            titleFont: { size: 12 },
            bodyFont:  { size: 12 },
          },
        },
        animation: { duration: 600, easing: 'easeInOutQuart' },
      },
    });

    return () => {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
    };
    // `total` se deriva de `data`, que ya es la dependencia
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 16px', marginBottom: 16 }}>
        {NAMES.map((name, i) => {
          const val = data[i]?.total || 0;
          const pct = total > 0 ? ((val / total) * 100).toFixed(1) : '0.0';
          return (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: T.textMid }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: COLORS[i], flexShrink: 0 }} />
              <span style={{ fontWeight: 500, color: T.text }}>{name}</span>
              <span style={{ color: T.textDim }}>{val} ({pct}%)</span>
            </div>
          );
        })}
      </div>
      <div style={{ position: 'relative', height: 200 }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Distribución: Neumonía ${data[0]?.total||0}, COVID-19 ${data[1]?.total||0}, Tuberculosis ${data[2]?.total||0}`}
        />
        <div style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center', pointerEvents: 'none',
        }}>
          {total > 0 ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 800, color: T.text, letterSpacing: '-0.02em', lineHeight: 1 }}>
                {total}
              </div>
              <div style={{ fontSize: 10, color: T.textDim, marginTop: 3, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                estudios
              </div>
            </>
          ) : (
            <div style={{ fontSize: 12, color: T.textDim }}>Sin datos</div>
          )}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ value, color }) {
  const pct = Math.min(value, 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
      <div style={{ flex: 1, height: 5, background: T.border, borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width 0.5s ease' }} />
      </div>
      <span style={{ fontSize: 11, fontWeight: 700, color, minWidth: 36, textAlign: 'right' }}>
        {value}%
      </span>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   DASHBOARD
══════════════════════════════════════════════════════════════════════════════*/
const DISEASE_COLORS = { pneumonia: T.primary, covid: T.danger, tuberculosis: T.warn };
const DISEASE_NAMES  = { pneumonia: 'Neumonía', covid: 'COVID-19', tuberculosis: 'Tuberculosis' };

function Dashboard() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    if (!window.Chart) {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
      document.head.appendChild(script);
    }
  }, []);

  useEffect(() => {
    authFetch(`${API_URL}/dashboard/stats`)
      .then(r => r.json())
      .then(data => { setStats(data); setLoading(false); })
      .catch(e  => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 80, gap: 12, color: T.textMid }}>
      <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} /> Cargando estadísticas...
    </div>
  );
  if (error) return (
    <div style={{ padding: '12px 16px', borderRadius: 8, background: T.dangerLt, color: T.danger, display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
      <AlertCircle size={14} /> Error: {error}
    </div>
  );
  if (!stats) return null;

  const { kpis, disease_distribution, timeline, system_metrics } = stats;

  const timelineShort  = timeline.slice(-14);
  const validationPct  = kpis.total_predictions > 0
    ? Math.round((kpis.validated / kpis.total_predictions) * 100)
    : 0;

  return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>

      {/* Encabezado */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 700, color: T.text, margin: 0, letterSpacing: '-0.01em' }}>
            Dashboard del Sistema
          </h2>
          <p style={{ fontSize: 12, color: T.textDim, margin: '4px 0 0' }}>
            Vista general de actividad y métricas del modelo
          </p>
        </div>
        <span style={badge(T.success, T.successLt)}>
          <CheckCircle size={10} /> Sistema activo
        </span>
      </div>

      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KpiCard Icon={Hospital}    label="Estudios totales"  value={kpis.total_studies}      color={T.primary} />
        <KpiCard Icon={Cpu}         label="Predicciones"      value={kpis.total_predictions}  color="#8b5cf6" />
        <KpiCard Icon={CheckCircle} label="Validados"         value={kpis.validated}          color={T.success}
          subtitle={`${validationPct}% del total`} />
        <KpiCard Icon={Clock}       label="Sin validar"       value={kpis.pending_validation} color={T.warn} />
        <KpiCard Icon={Layers}      label="Estudios DICOM"    value={kpis.dicom_studies}      color="#0ea5e9" />
      </div>

      {/* Métricas del modelo */}
      <div style={card({ background: T.primaryLt, border: `1px solid #c7d7f9`, marginBottom: 20 })}>
        <div style={{ fontSize: 11, fontWeight: 700, color: T.primary, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 14 }}>
          Métricas de Validación del Modelo (ensemble)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '12px 24px' }}>
          {[
            { label: 'Neumonía',     value: system_metrics.pneumonia_accuracy,   note: 'AUC 0.972' },
            { label: 'COVID-19',     value: system_metrics.covid_accuracy,        note: 'AUC 1.000 ⚠️' },
            { label: 'Tuberculosis', value: system_metrics.tuberculosis_accuracy, note: 'AUC 1.000 ⚠️' },
            { label: 'Promedio',     value: system_metrics.avg_accuracy,          note: 'ensemble' },
            { label: 'YOLO mAP50',  value: system_metrics.yolo_map50,            note: 'localiza lesiones',
              hint: 'mAP50: qué tan bien YOLO localiza lesiones en la radiografía (1.0 = perfecto).' },
            { label: 'U-Net Dice',  value: system_metrics.unet_dice,             note: 'contorno pulmonar',
              hint: 'Dice: qué tanto coincide el contorno de pulmón dibujado por U-Net con el real (1.0 = idéntico).' },
          ].map(({ label, value, note, hint }) => (
            <div key={label} title={hint}>
              <div style={{ fontSize: 20, fontWeight: 800, color: T.primary, letterSpacing: '-0.02em' }}>{value}</div>
              <div style={{ fontSize: 11, color: T.textMid, marginTop: 2, fontWeight: 500 }}>{label}</div>
              <div style={{ fontSize: 10, color: T.textDim, marginTop: 1 }}>{note}</div>
            </div>
          ))}
        </div>
        <div style={{
          marginTop: 14, fontSize: 11, color: T.primaryDk,
          background: '#dbeafe', borderRadius: 6, padding: '6px 10px', display: 'inline-block',
        }}>
          ⚠️ COVID y TB al 100% — interpretar con cautela (posible sobreajuste / n reducido en prueba)
        </div>
      </div>

      {/* Gráficas */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 14, flexWrap: 'wrap' }}>

        {/* Donut */}
        <div style={card({ flex: 1, minWidth: 280, marginBottom: 0 })}>
          <div style={{ fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 16 }}>
            Distribución por enfermedad
          </div>
          {kpis.total_predictions > 0
            ? <DonutChart data={disease_distribution} />
            : <div style={{ textAlign: 'center', padding: 40, color: T.textDim, fontSize: 13 }}>Sin datos aún</div>
          }
        </div>

        {/* Confianza con MiniBar */}
        <div style={card({ flex: 1, minWidth: 280, marginBottom: 0 })}>
          <div style={{ fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 16 }}>
            Confianza y tasa positiva
          </div>
          {kpis.total_predictions > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              {disease_distribution.map((d) => {
                const color = DISEASE_COLORS[d.disease];
                const name  = DISEASE_NAMES[d.disease];
                return (
                  <div key={d.disease}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                      <span style={{ fontSize: 12, fontWeight: 600, color: T.text }}>{name}</span>
                      <span style={{ fontSize: 11, color: T.textDim, marginLeft: 'auto' }}>
                        {d.total} estudio{d.total !== 1 ? 's' : ''}
                      </span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: T.textMid }}>
                        <span style={{ minWidth: 84 }}>Confianza</span>
                        <MiniBar value={d.avg_confidence} color={color} />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: T.textMid }}>
                        <span style={{ minWidth: 84 }}>Tasa positiva</span>
                        <MiniBar value={d.positive_rate} color={color + '99'} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: T.textDim, fontSize: 13 }}>Sin datos aún</div>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div style={card({ marginBottom: 14 })}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: T.text }}>Estudios — últimos 14 días</div>
          <div style={{ fontSize: 11, color: T.textDim }}>
            Total periodo: {timelineShort.reduce((s, d) => s + (d.count || 0), 0)}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={timelineShort} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="gradDash" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={T.primary} stopOpacity={0.15} />
                <stop offset="95%" stopColor={T.primary} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={T.borderDim} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: T.textDim }}
              tickFormatter={d => String(d).replace('Z','').slice(5, 10)}
            />
            <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: T.textDim }} />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,31,56,0.92)', border: 'none',
                borderRadius: 8, fontSize: 12, color: '#fff', padding: '8px 12px',
              }}
              labelFormatter={d => `Fecha: ${String(d).replace('Z','').slice(0,10)}`}
              formatter={v => [`${v} estudio${v !== 1 ? 's' : ''}`, '']}
            />
            <Area
              type="monotone" dataKey="count"
              stroke={T.primary} fill="url(#gradDash)"
              strokeWidth={2} dot={{ fill: T.primary, r: 3 }}
              activeDot={{ r: 5, stroke: '#fff', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Tabla detallada */}
      <div style={card()}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 14 }}>
          Resumen por enfermedad
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 500 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${T.border}` }}>
                {['Enfermedad','Total','Positivos','Negativos','Tasa positiva','Confianza prom.'].map(h => (
                  <th key={h} style={{
                    padding: '8px 12px', textAlign: 'left', color: T.textMid,
                    fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {disease_distribution.map((d) => {
                const color  = DISEASE_COLORS[d.disease];
                const name   = DISEASE_NAMES[d.disease];
                const isRisk = d.positive_rate > 50;
                return (
                  <tr key={d.disease} style={{ borderBottom: `1px solid ${T.borderDim}` }}>
                    <td style={{ padding: '12px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                        <span style={{ fontWeight: 600, color: T.text }}>{name}</span>
                      </div>
                    </td>
                    <td style={{ padding: '12px 12px', fontWeight: 700, color: T.text }}>{d.total}</td>
                    <td style={{ padding: '12px 12px', color: T.danger, fontWeight: 600 }}>{d.positive}</td>
                    <td style={{ padding: '12px 12px', color: T.success, fontWeight: 600 }}>{d.negative}</td>
                    <td style={{ padding: '12px 12px' }}>
                      <span style={badge(isRisk ? T.danger : T.success, isRisk ? T.dangerLt : T.successLt)}>
                        {d.positive_rate}%
                      </span>
                    </td>
                    <td style={{ padding: '12px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 48, height: 4, background: T.border, borderRadius: 2, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${d.avg_confidence}%`, background: color, borderRadius: 2 }} />
                        </div>
                        <span style={{ fontWeight: 700, color, fontSize: 12 }}>{d.avg_confidence}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   HISTORIAL
══════════════════════════════════════════════════════════════════════════════*/
function History() {
  const [studies,    setStudies]    = useState([]);
  const [loading,    setLoading]    = useState(true);
  const [disease,    setDisease]    = useState('all');
  const [resultado,  setResultado]  = useState('all');
  const [validacion, setValidacion] = useState('all');
  const [fechaDesde, setFechaDesde] = useState('');
  const [fechaHasta, setFechaHasta] = useState('');
  const [error,      setError]      = useState(null);

  const load = useCallback(() => {
    setLoading(true); setError(null);
    authFetch(`${API_URL}/studies`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setStudies(extractStudies(data)); setLoading(false); })
      .catch(() => { setError('No se pudo cargar el historial. Revisa que el servidor esté activo.'); setLoading(false); });
  }, []);
  useEffect(() => { load(); }, [load]);

  const downloadPDF = (studyId) => {
    authFetch(`${API_URL}/studies/${studyId}/report`)
      .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `reporte_${studyId.slice(0,8)}.pdf`; a.click();
        URL.revokeObjectURL(url);
      })
      .catch(() => setError('No se pudo descargar el PDF. Inténtalo de nuevo.'));
  };

  const filtered = studies.filter(s => {
    const pred   = s.predictions?.[0];
    const isPos  = pred && !pred.prediction?.toLowerCase().includes('normal');
    const dtype  = String(s.disease_type).toUpperCase().replace('DISEASETYPE.', '');
    const fecha  = new Date(s.created_at);

    if (disease !== 'all') {
      if (disease === 'pneumonia'    && !dtype.includes('PNEUMONIA'))    return false;
      if (disease === 'covid'        && !dtype.includes('COVID'))        return false;
      if (disease === 'tuberculosis' && !dtype.includes('TUBERCUL'))     return false;
    }
    if (resultado  === 'positive' && !isPos)           return false;
    if (resultado  === 'normal'   && isPos)            return false;
    if (validacion === 'validated' && !pred?.is_validated) return false;
    if (validacion === 'pending'   && pred?.is_validated)  return false;
    if (fechaDesde && fecha < new Date(fechaDesde))    return false;
    if (fechaHasta && fecha > new Date(fechaHasta + 'T23:59:59')) return false;
    return true;
  });

  const inputStyle = {
    padding: '6px 10px', borderRadius: 7, border: `1px solid ${T.border}`,
    fontSize: 12, fontFamily: T.font, color: T.text, background: T.surface,
    outline: 'none', cursor: 'pointer',
  };

  const selectStyle = {
    ...inputStyle, cursor: 'pointer',
  };

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 60, color: T.textMid, fontFamily: T.font }}>
      <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} /> Cargando historial...
    </div>
  );

  if (error && studies.length === 0) return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>
      <ErrorState message={error} onRetry={load} />
    </div>
  );

const exportCSV = () => {
  const headers = ['ID', 'Enfermedad', 'Resultado', 'Confianza', 'Validado', 'Fecha'];
  const rows = filtered.map(s => {
    const pred   = s.predictions?.[0];
    const isPos  = pred && !pred.prediction?.toLowerCase().includes('normal');
    const dtype  = String(s.disease_type).toUpperCase().replace('DISEASETYPE.', '');
    const dlabel = dtype.includes('PNEUMONIA') ? 'Neumonía'
      : dtype.includes('COVID') ? 'COVID-19'
      : dtype.includes('TUBERCUL') ? 'Tuberculosis'
      : s.disease_type;
    return [
      s.id,
      dlabel,
      isPos ? 'Positivo' : 'Normal',
      pred ? `${(pred.confidence * 100).toFixed(1)}%` : '—',
      pred?.is_validated ? 'Sí' : 'No',
      fmtDate(s.created_at),
    ];
  });

  const csv = [headers, ...rows]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `historial_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
};



  return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: T.text, margin: 0, letterSpacing: '-0.01em' }}>
          Historial de Estudios
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 12, color: T.textDim }}>
            {filtered.length} de {studies.length} registro{studies.length !== 1 ? 's' : ''}
          </span>
          {filtered.length > 0 && (
            <button onClick={exportCSV} style={btn('ghost', { fontSize: 12, padding: '5px 12px' })}>
              <Download size={12} /> Exportar CSV
            </button>
          )}
        </div>
      </div>

      {error && <div style={{ marginBottom: 12 }}><ErrorState message={error} /></div>}

      {/* Filtros */}
      <div style={{
        ...card({ marginBottom: 16 }),
        display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-end',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Enfermedad
          </label>
          <select value={disease} onChange={e => setDisease(e.target.value)} style={selectStyle}>
            <option value="all">Todas</option>
            <option value="pneumonia">Neumonía</option>
            <option value="covid">COVID-19</option>
            <option value="tuberculosis">Tuberculosis</option>
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Resultado
          </label>
          <select value={resultado} onChange={e => setResultado(e.target.value)} style={selectStyle}>
            <option value="all">Todos</option>
            <option value="positive">Positivo</option>
            <option value="normal">Normal</option>
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Validación
          </label>
          <select value={validacion} onChange={e => setValidacion(e.target.value)} style={selectStyle}>
            <option value="all">Todas</option>
            <option value="validated">Validado</option>
            <option value="pending">Sin validar</option>
          </select>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Desde
          </label>
          <input type="date" value={fechaDesde} onChange={e => setFechaDesde(e.target.value)} style={inputStyle} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Hasta
          </label>
          <input type="date" value={fechaHasta} onChange={e => setFechaHasta(e.target.value)} style={inputStyle} />
        </div>

        {(disease !== 'all' || resultado !== 'all' || validacion !== 'all' || fechaDesde || fechaHasta) && (
          <button
            onClick={() => { setDisease('all'); setResultado('all'); setValidacion('all'); setFechaDesde(''); setFechaHasta(''); }}
            style={btn('ghost', { fontSize: 12, padding: '6px 12px' })}
          >
            <X size={12} /> Limpiar filtros
          </button>
        )}
      </div>

      {/* Lista */}
      {filtered.length === 0 ? (
        <div style={{ ...card(), textAlign: 'center', padding: 60, color: T.textDim }}>
          <ClipboardList size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
          <p style={{ margin: 0 }}>
            {studies.length === 0 ? 'No hay estudios registrados aún' : 'Ningún estudio coincide con los filtros'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {filtered.map(study => {
            const pred      = study.predictions?.[0];
            const isPos     = pred && !pred.prediction?.toLowerCase().includes('normal');
            const dtype     = String(study.disease_type).toUpperCase().replace('DISEASETYPE.', '');
            const diseaseLabel = dtype.includes('PNEUMONIA') ? 'Neumonía'
              : dtype.includes('COVID') ? 'COVID-19'
              : dtype.includes('TUBERCUL') ? 'Tuberculosis'
              : study.disease_type;
            return (
              <div key={study.id} style={{
                ...card({ marginBottom: 0, display: 'flex', gap: 16, alignItems: 'center',
                  borderLeft: `4px solid ${isPos ? T.danger : T.success}` }),
              }}>
                {study.original_image_url && (
                  <img src={study.original_image_url} alt="rx" style={{
                    width: 60, height: 60, borderRadius: 8, objectFit: 'cover',
                    border: `1px solid ${T.border}`, flexShrink: 0,
                  }} />
                )}
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ fontWeight: 600, fontSize: 14, color: T.text }}>{diseaseLabel}</div>
                    <div style={{ fontSize: 11, color: T.textDim }}>{fmtDate(study.created_at)}</div>
                  </div>
                  {pred && (
                    <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                      <span style={badge(isPos ? T.danger : T.success, isPos ? T.dangerLt : T.successLt)}>
                        {isPos ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
                        {isPos ? 'Positivo' : 'Normal'}
                      </span>
                      <span style={{ fontSize: 12, color: T.textMid }}>
                        Confianza: <strong style={{ color: T.text }}>{(pred.confidence * 100).toFixed(1)}%</strong>
                      </span>
                      {pred.is_validated && (
                        <span style={badge(T.success, T.successLt)}><Check size={10} />Validado</span>
                      )}
                      <button onClick={() => downloadPDF(study.id)} style={{
                        ...btn('ghost', { padding: '3px 10px', fontSize: 11 }), marginLeft: 'auto',
                      }}>
                        <Download size={11} /> PDF
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   PACIENTES
══════════════════════════════════════════════════════════════════════════════*/
function Patients() {
  const [studies,   setStudies]   = useState([]);
  const [loading,   setLoading]   = useState(true);
  const [selected,  setSelected]  = useState(null);
  const [busqueda,  setBusqueda]  = useState('');
  const [error,     setError]     = useState(null);

  const load = useCallback(() => {
    setLoading(true); setError(null);
    authFetch(`${API_URL}/studies`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setStudies(extractStudies(data)); setLoading(false); })
      .catch(() => { setError('No se pudo cargar la lista de pacientes. Revisa que el servidor esté activo.'); setLoading(false); });
  }, []);
  useEffect(() => { load(); }, [load]);

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 60, color: T.textMid, fontFamily: T.font }}>
      <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} /> Cargando pacientes...
    </div>
  );

  if (error) return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>
      <ErrorState message={error} onRetry={load} />
    </div>
  );

  // Agrupar por paciente
  const patients = {};
  studies.forEach(s => {
    const key = s.patient_identifier || 'Sin identificador';
    if (!patients[key]) patients[key] = [];
    patients[key].push(s);
  });

  // Filtrar por búsqueda
  const patientKeys = Object.keys(patients).filter(name =>
    name.toLowerCase().includes(busqueda.toLowerCase())
  );

  return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: T.text, margin: 0, letterSpacing: '-0.01em' }}>
          Pacientes
        </h2>
        <span style={{ fontSize: 12, color: T.textDim }}>
          {patientKeys.length} de {Object.keys(patients).length} paciente{Object.keys(patients).length !== 1 ? 's' : ''}
        </span>
      </div>

      {Object.keys(patients).length === 0 ? (
        <div style={{ ...card(), textAlign: 'center', padding: 60, color: T.textDim }}>
          <Users size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
          <p style={{ margin: 0 }}>No hay pacientes registrados aún</p>
        </div>
      ) : (
        <div className="patients-layout">
          {/* Lista de pacientes */}
          <div className="patients-list" style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>

            {/* Buscador */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 12px', borderRadius: 8,
              border: `1px solid ${T.border}`, background: T.surface,
              marginBottom: 4,
            }}>
              <Search size={13} color={T.textDim} />
              <input
                type="text"
                placeholder="Buscar paciente..."
                value={busqueda}
                onChange={e => { setBusqueda(e.target.value); setSelected(null); }}
                style={{
                  border: 'none', outline: 'none', fontSize: 12,
                  fontFamily: T.font, color: T.text, background: 'transparent',
                  width: '100%',
                }}
              />
              {busqueda && (
                <button
                  onClick={() => { setBusqueda(''); setSelected(null); }}
                  style={{ border: 'none', background: 'none', cursor: 'pointer', padding: 0, display: 'flex' }}
                >
                  <X size={12} color={T.textDim} />
                </button>
              )}
            </div>

            {patientKeys.length === 0 ? (
              <div style={{ fontSize: 12, color: T.textDim, textAlign: 'center', padding: '20px 0' }}>
                Sin resultados
              </div>
            ) : (
              patientKeys.map(name => (
                <button key={name} onClick={() => setSelected(name)} style={{
                  ...card({ marginBottom: 0, cursor: 'pointer', textAlign: 'left',
                    borderLeft: selected === name ? `4px solid ${T.primary}` : `4px solid transparent`,
                    background: selected === name ? T.primaryLt : T.surface,
                  }),
                  border: `1px solid ${selected === name ? '#c7d7f9' : T.border}`,
                  fontFamily: T.font,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: '50%',
                      background: T.primaryLt, display: 'flex', alignItems: 'center', justifyContent: 'center',
                      flexShrink: 0,
                    }}>
                      <Users size={14} color={T.primary} />
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                      <div style={{
                        fontSize: 13, fontWeight: 600, color: T.text,
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      }}>
                        {name}
                      </div>
                      <div style={{ fontSize: 11, color: T.textDim }}>
                        {patients[name].length} estudio{patients[name].length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>

          {/* Detalle del paciente seleccionado */}
          {selected && patients[selected] ? (
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={card({ background: T.primaryLt, border: `1px solid #c7d7f9`, marginBottom: 12 })}>
                <div style={{ fontWeight: 700, fontSize: 15, color: T.primary }}>{selected}</div>
                <div style={{ fontSize: 12, color: T.textMid, marginTop: 4 }}>
                  {patients[selected].length} estudio{patients[selected].length !== 1 ? 's' : ''} en el sistema
                </div>
              </div>
              {patients[selected]
                .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
                .map(s => {
                  const pred  = s.predictions?.[0];
                  const isPos = pred && !pred.prediction?.toLowerCase().includes('normal');
                  const dtype = String(s.disease_type).toUpperCase().replace('DISEASETYPE.', '');
                  const dlabel = dtype.includes('PNEUMONIA') ? 'Neumonía'
                    : dtype.includes('COVID') ? 'COVID-19'
                    : dtype.includes('TUBERCUL') ? 'Tuberculosis'
                    : s.disease_type;
                  return (
                    <div key={s.id} style={card({
                      borderLeft: `4px solid ${isPos ? T.danger : T.success}`, marginBottom: 8,
                    })}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: T.text }}>{dlabel}</span>
                        <span style={{ fontSize: 11, color: T.textDim }}>{fmtShort(s.created_at)}</span>
                      </div>
                      {pred && (
                        <div style={{ marginTop: 6, display: 'flex', gap: 8, alignItems: 'center' }}>
                          <span style={badge(isPos ? T.danger : T.success, isPos ? T.dangerLt : T.successLt)}>
                            {isPos ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
                            {isPos ? 'Positivo' : 'Normal'}
                          </span>
                          <span style={{ fontSize: 12, color: T.textMid }}>
                            Confianza: <strong>{(pred.confidence * 100).toFixed(1)}%</strong>
                          </span>
                          {pred.is_validated && (
                            <span style={badge(T.success, T.successLt)}><Check size={10} />Validado</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          ) : (
            <div style={{ flex: 1, ...card({ textAlign: 'center', padding: 60, color: T.textDim }) }}>
              <Users size={36} style={{ marginBottom: 12, opacity: 0.2 }} />
              <p style={{ margin: 0, fontSize: 13 }}>Selecciona un paciente para ver sus estudios</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   VALIDATION MODAL
══════════════════════════════════════════════════════════════════════════════*/
function ValidationModal({ study, pred, onClose, onSaved }) {
  const [result, setResult]   = useState('confirmed');
  const [notes, setNotes]     = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const isPos = pred && !pred.prediction?.toLowerCase().includes('normal');
  const diseaseLabel = { PNEUMONIA: 'Neumonía', COVID: 'COVID-19', TUBERCULOSIS: 'Tuberculosis' }[study.disease_type] || study.disease_type;

  const handleSubmit = async () => {
    setLoading(true); setError(null);
    try {
      const params = new URLSearchParams({ validation_result: result });
      if (notes.trim()) params.append('radiologist_notes', notes.trim());
      const res = await authFetch(`${API_URL}/predictions/${pred.id}/validate?${params}`, { method: 'PUT' });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Error al validar'); }
      onSaved();
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" style={{
      position: 'fixed', inset: 0, background: 'rgba(15,31,56,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
      backdropFilter: 'blur(4px)', fontFamily: T.font,
    }} onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-panel" role="dialog" aria-modal="true" aria-label="Validar estudio" style={{
        background: T.surface, borderRadius: T.radiusLg, padding: 28,
        width: '90%', maxWidth: 580, maxHeight: '90vh', overflowY: 'auto',
        boxShadow: '0 24px 64px rgba(0,0,0,0.2)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: T.text }}>Validar Estudio</h3>
            <p style={{ margin: '2px 0 0', fontSize: 12, color: T.textMid }}>{diseaseLabel}</p>
          </div>
          <button onClick={onClose} aria-label="Cerrar" style={{ ...btn('ghost', { padding: 8, borderRadius: '50%' }) }}>
            <X size={16} />
          </button>
        </div>

        {/* Study info */}
        <div style={card({ display: 'flex', gap: 14, marginBottom: 16 })}>
          {study.original_image_url && (
            <img src={study.original_image_url} alt="rx" style={{
              width: 72, height: 72, borderRadius: 8, objectFit: 'cover',
              border: `1px solid ${T.border}`, flexShrink: 0,
            }} />
          )}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: T.textMid, marginBottom: 6 }}>{fmtDate(study.created_at)}</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
              <span style={badge(isPos ? T.danger : T.success, isPos ? T.dangerLt : T.successLt)}>
                {isPos ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
                {isPos ? 'IA: Positivo' : 'IA: Normal'}
              </span>
            </div>
            {pred && (
              <div style={{ fontSize: 12, color: T.textMid }}>
                Confianza IA: <strong style={{ color: T.primary }}>{(pred.confidence * 100).toFixed(1)}%</strong>
              </div>
            )}
          </div>
        </div>

        {/* Images */}
        {pred && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 16, overflowX: 'auto' }}>
            {[
              [pred.gradcam_image_url, 'Grad-CAM'],
              [pred.yolo_image_url, 'YOLO'],
              [pred.segmentation_image_url, 'U-Net'],
            ].filter(([url]) => url).map(([url, label]) => (
              <div key={label} style={{ textAlign: 'center', flexShrink: 0 }}>
                <img src={url} alt={label} style={{
                  width: 110, height: 90, objectFit: 'cover', borderRadius: 8,
                  border: `1px solid ${T.border}`,
                }} />
                <div style={{ fontSize: 10, color: T.textDim, marginTop: 4, fontWeight: 600 }}>{label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Decision */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: T.textMid, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10 }}>
            Decisión del radiólogo
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {[
              { value: 'confirmed',    label: 'Confirmar',    Icon: Check,       color: T.success, bg: T.successLt },
              { value: 'rejected',     label: 'Rechazar IA',  Icon: X,           color: T.danger,  bg: T.dangerLt },
              { value: 'inconclusive', label: 'Inconclusivo', Icon: HelpCircle,  color: T.warn,    bg: T.warnLt },
            ].map(opt => (
              <button key={opt.value} onClick={() => setResult(opt.value)} style={{
                flex: 1, padding: '10px 8px', borderRadius: 8, cursor: 'pointer',
                fontSize: 12, fontWeight: 700, fontFamily: T.font,
                border: `2px solid ${result === opt.value ? opt.color : T.border}`,
                background: result === opt.value ? opt.bg : T.surface,
                color: result === opt.value ? opt.color : T.textMid,
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}>
                <opt.Icon size={13} />{opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Notes */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: T.textMid, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            Notas <span style={{ fontWeight: 400, opacity: 0.6 }}>(opcional)</span>
          </div>
          <textarea value={notes} onChange={e => setNotes(e.target.value)}
            placeholder="Hallazgos clínicos, recomendaciones..."
            rows={3}
            style={{
              width: '100%', padding: '10px 12px', borderRadius: 8, fontSize: 13,
              border: `1px solid ${T.border}`, resize: 'vertical', fontFamily: T.font,
              boxSizing: 'border-box', outline: 'none', color: T.text,
            }} />
        </div>

        {error && (
          <div style={{ ...card({ background: T.dangerLt, border: `1px solid #fecaca`, marginBottom: 12 }), color: T.danger, display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
            <AlertCircle size={14} />{error}
          </div>
        )}

        <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={btn('ghost')}>Cancelar</button>
          <button onClick={handleSubmit} disabled={loading} style={btn('primary', { opacity: loading ? 0.6 : 1 })}>
            {loading ? <><RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} />Guardando...</> : <><Check size={13} />Guardar validación</>}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   PANEL RADIÓLOGO
══════════════════════════════════════════════════════════════════════════════*/
function RadiologistPanel() {
  const [studies, setStudies]   = useState([]);
  const [loading, setLoading]   = useState(true);
  const [filter, setFilter]     = useState('pending');
  const [selected, setSelected] = useState(null);
  const [tab, setTab]           = useState('queue');

  const diseaseLabel = (t) => ({ PNEUMONIA: 'Neumonía', COVID: 'COVID-19', TUBERCULOSIS: 'Tuberculosis' }[t] ||
    ({ PNEUMONIA: 'Neumonía', COVID: 'COVID-19', TUBERCULOSIS: 'Tuberculosis' }[String(t).toUpperCase()]) || String(t));

  const [error, setError]       = useState(null);
  const closeModal              = useCallback(() => setSelected(null), []);

  // `silent`: refresco en segundo plano; no reemplaza la lista por el cargador
  const loadStudies = useCallback((silent = false) => {
    if (!silent) { setLoading(true); setError(null); }
    authFetch(`${API_URL}/studies`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { setStudies(extractStudies(data)); setError(null); setLoading(false); })
      .catch(() => {
        if (!silent) setError('No se pudieron cargar los estudios. Revisa que el servidor esté activo.');
        setLoading(false);
      });
  }, []);
  useEffect(() => {
    loadStudies();
    const interval = setInterval(() => loadStudies(true), 30000);
    return () => clearInterval(interval);
  }, [loadStudies]);

  const pendingCount   = studies.filter(s => s.predictions?.[0] && !s.predictions[0].is_validated).length;
  const validatedCount = studies.filter(s => s.predictions?.[0]?.is_validated).length;
  const filtered = studies.filter(s => {
    const p = s.predictions?.[0];
    if (!p) return false;
    if (filter === 'pending')   return !p.is_validated;
    if (filter === 'validated') return p.is_validated;
    return true;
  });

  const auditEntries = studies
    .filter(s => s.predictions?.[0]?.is_validated)
    .map(s => {
      const p = s.predictions[0];
      return { study_id: s.id, disease: diseaseLabel(s.disease_type), result: p.validation_result,
        notes: p.radiologist_notes, validated_at: p.validated_at, prediction: p.prediction, confidence: p.confidence };
    })
    .sort((a, b) => new Date(b.validated_at) - new Date(a.validated_at));

  const RESULT = {
    confirmed:    { color: T.success, bg: T.successLt, label: 'Confirmado',  Icon: CheckCircle },
    rejected:     { color: T.danger,  bg: T.dangerLt,  label: 'Rechazado',   Icon: X },
    inconclusive: { color: T.warn,    bg: T.warnLt,    label: 'Inconclusivo',Icon: HelpCircle },
  };

  return (
    <div style={{ paddingBottom: 32, fontFamily: T.font }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: T.text, margin: 0, letterSpacing: '-0.01em' }}>Panel Radiólogo</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <span style={badge(T.warn, T.warnLt)}><Clock size={10} />{pendingCount} pendientes</span>
          <span style={badge(T.success, T.successLt)}><CheckCircle size={10} />{validatedCount} validados</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: `2px solid ${T.border}` }}>
        {[{ key: 'queue', label: 'Cola de revisión', Icon: ClipboardList }, { key: 'audit', label: 'Audit trail', Icon: Eye }].map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '10px 16px', border: 'none', cursor: 'pointer', fontSize: 13,
            fontWeight: tab === t.key ? 700 : 400, fontFamily: T.font,
            background: 'transparent',
            borderBottom: tab === t.key ? `2px solid ${T.primary}` : '2px solid transparent',
            color: tab === t.key ? T.primary : T.textMid, marginBottom: -2,
          }}>
            <t.Icon size={14} />{t.label}
          </button>
        ))}
      </div>

      {tab === 'queue' && (
        <>
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            {[
              { key: 'pending',   label: `Pendientes (${pendingCount})` },
              { key: 'validated', label: `Validados (${validatedCount})` },
              { key: 'all',       label: 'Todos' },
            ].map(f => (
              <button key={f.key} onClick={() => setFilter(f.key)} style={{
                padding: '6px 14px', borderRadius: 20, border: 'none', cursor: 'pointer',
                fontSize: 12, fontWeight: 600, fontFamily: T.font,
                background: filter === f.key ? T.primary : T.border,
                color: filter === f.key ? '#fff' : T.textMid,
              }}>{f.label}</button>
            ))}
          </div>

          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 60, color: T.textMid }}>
              <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />Cargando estudios...
            </div>
          ) : error ? (
            <ErrorState message={error} onRetry={() => loadStudies()} />
          ) : filtered.length === 0 ? (
            <div style={{ ...card(), textAlign: 'center', padding: 60, color: T.textDim }}>
              {filter === 'pending' ? <CheckCircle size={40} style={{ marginBottom: 12, opacity: 0.3, color: T.success }} /> : <ClipboardList size={40} style={{ marginBottom: 12, opacity: 0.3 }} />}
              <p style={{ margin: 0 }}>{filter === 'pending' ? 'Sin estudios pendientes' : 'Sin estudios en este filtro'}</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {filtered.map(study => {
                const pred        = study.predictions?.[0];
                if (!pred) return null;
                const isPos       = !pred.prediction?.toLowerCase().includes('normal');
                const isValidated = pred.is_validated;
                const res         = isValidated ? (RESULT[pred.validation_result] || RESULT.inconclusive) : null;

                return (
                  <div key={study.id} style={card({
                    marginBottom: 0, display: 'flex', gap: 14, alignItems: 'center',
                    borderLeft: `4px solid ${isValidated ? T.success : (isPos ? T.danger : T.primary)}`,
                  })}>
                    {study.original_image_url && (
                      <img src={study.original_image_url} alt="rx" style={{
                        width: 64, height: 64, borderRadius: 8, objectFit: 'cover',
                        border: `1px solid ${T.border}`, flexShrink: 0,
                      }} />
                    )}
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 14, color: T.text }}>{diseaseLabel(study.disease_type)}</div>
                          <div style={{ fontSize: 11, color: T.textDim, marginTop: 2 }}>
                            ID: {study.id.slice(0,8)}… · {fmtDate(study.created_at)}
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                          {isValidated && res && (
                            <span style={badge(res.color, res.bg)}><res.Icon size={10} />{res.label}</span>
                          )}
                          {isValidated && (
                            <button onClick={() => exportFHIR(study, pred)} style={btn('ghost', { fontSize: 11, padding: '4px 10px' })}>
                              <Share2 size={11} /> FHIR
                            </button>
                          )}
                          {!isValidated && (
                            <button onClick={() => setSelected({ study, pred })} style={btn('primary', { fontSize: 12 })}>
                              Validar <ChevronRight size={13} />
                            </button>
                          )}
                        </div>
                      </div>
                      <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <span style={badge(isPos ? T.danger : T.success, isPos ? T.dangerLt : T.successLt)}>
                          {isPos ? <AlertTriangle size={10} /> : <CheckCircle size={10} />}
                          {isPos ? 'IA: Positivo' : 'IA: Normal'}
                        </span>
                        <span style={{ fontSize: 12, color: T.textMid }}>
                          Confianza: <strong style={{ color: T.text }}>{(pred.confidence * 100).toFixed(1)}%</strong>
                        </span>
                        {isValidated && pred.radiologist_notes && (
                          <span style={{ fontSize: 11, color: T.primary, fontStyle: 'italic' }}>
                            "{pred.radiologist_notes.slice(0, 60)}{pred.radiologist_notes.length > 60 ? '…' : ''}"
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {tab === 'audit' && (
        <div>
          {auditEntries.length === 0 ? (
            <div style={{ ...card(), textAlign: 'center', padding: 60, color: T.textDim }}>
              <Eye size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
              <p style={{ margin: 0 }}>No hay validaciones registradas aún</p>
            </div>
          ) : (
            <>
              <div style={{ fontSize: 12, color: T.textDim, marginBottom: 12 }}>
                {auditEntries.length} validación{auditEntries.length !== 1 ? 'es' : ''} registrada{auditEntries.length !== 1 ? 's' : ''}
              </div>
              <div style={{ border: `1px solid ${T.border}`, borderRadius: T.radius, overflowX: 'auto', background: T.surface }}>
                <table style={{ width: '100%', minWidth: 640, borderCollapse: 'collapse', fontSize: 13, fontFamily: T.font }}>
                  <thead>
                    <tr style={{ background: T.bg }}>
                      {['Fecha', 'Enfermedad', 'Pred. IA', 'Confianza', 'Decisión', 'Notas'].map(h => (
                        <th key={h} style={{ padding: '10px 14px', textAlign: 'left', color: T.textMid,
                          fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em',
                          borderBottom: `1px solid ${T.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {auditEntries.map((entry, i) => {
                      const res   = RESULT[entry.result] || RESULT.inconclusive;
                      const isPos = !entry.prediction?.toLowerCase().includes('normal');
                      return (
                        <tr key={entry.study_id + i} style={{ borderBottom: `1px solid ${T.borderDim}` }}>
                          <td style={{ padding: '10px 14px', color: T.textMid, fontSize: 12, whiteSpace: 'nowrap' }}>
                            {entry.validated_at ? fmtDate(entry.validated_at) : '—'}
                          </td>
                          <td style={{ padding: '10px 14px', fontWeight: 600 }}>{entry.disease}</td>
                          <td style={{ padding: '10px 14px' }}>
                            <span style={badge(isPos ? T.danger : T.success, isPos ? T.dangerLt : T.successLt)}>
                              {isPos ? 'Positivo' : 'Normal'}
                            </span>
                          </td>
                          <td style={{ padding: '10px 14px', fontWeight: 700, color: T.primary }}>
                            {(entry.confidence * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: '10px 14px' }}>
                            <span style={badge(res.color, res.bg)}><res.Icon size={10} />{res.label}</span>
                          </td>
                          <td style={{ padding: '10px 14px', color: T.primary, fontStyle: entry.notes ? 'italic' : 'normal',
                            maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {entry.notes || <span style={{ color: T.textDim }}>—</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {selected && (
        <ValidationModal
          study={selected.study} pred={selected.pred}
          onClose={closeModal}
          onSaved={() => { setSelected(null); loadStudies(true); }}
        />
      )}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   ANÁLISIS
══════════════════════════════════════════════════════════════════════════════*/
function ProbBar({ label, value, color }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4, color: T.textMid }}>
        <span>{label}</span>
        <span style={{ fontWeight: 700, color: T.text }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: T.border, borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 3, transition: 'width .4s' }} />
      </div>
    </div>
  );
}

function DiseaseCard({ item, index }) {
  const NAMES  = { pneumonia: 'Neumonía', covid: 'COVID-19', tuberculosis: 'Tuberculosis' };
  const COLORS = { pneumonia: T.primary, covid: T.danger, tuberculosis: T.warn };
  const ICONS  = { pneumonia: Wind, covid: Activity, tuberculosis: Microscope };
  const color  = COLORS[item.disease];
  const Icon   = ICONS[item.disease] || Activity;
  const isPos  = item.is_positive;

  return (
    <div className="reveal" style={card({ '--i': index + 1, borderLeft: `4px solid ${isPos ? color : T.success}`, opacity: index === 0 ? 1 : 0.8, marginBottom: 10 })}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, flexShrink: 0, background: color + '18', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon size={18} color={color} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: T.text }}>#{item.rank} {NAMES[item.disease]}</div>
            <span style={badge(isPos ? color : T.success, isPos ? color + '18' : T.successLt, { fontSize: 10 })}>
              {isPos ? <AlertTriangle size={9} /> : <CheckCircle size={9} />}
              {isPos ? 'Positivo' : 'Normal'}
            </span>
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{
            display: 'inline-block', padding: '6px 16px', borderRadius: 20,
            background: isPos ? color : T.success, color: '#fff',
            fontWeight: 800, fontSize: 16, letterSpacing: '-0.01em',
          }}>
            {(item.probability_disease * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: 10, color: T.textDim, marginTop: 4 }}>probabilidad de enfermedad</div>
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <ProbBar label="Confianza del modelo en este veredicto" value={item.confidence} color={isPos ? color : T.success} />
      </div>
      <details style={{ marginTop: 8 }}>
        <summary style={{ fontSize: 11, color: T.textDim, cursor: 'pointer', userSelect: 'none' }}>
          Ver detalle por modelo
        </summary>
        <div style={{ marginTop: 10, paddingLeft: 4 }}>
          {Object.entries(item.ensemble_detail).map(([model, probs]) => {
            const mName = model === 'densenet' ? 'DenseNet121' : model === 'resnet' ? 'ResNet50' : 'EfficientNet-B0';
            const classNames = Object.keys(item.probabilities);
            return (
              <div key={model} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: T.textMid, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{mName}</div>
                {probs.map((p, i) => <ProbBar key={i} label={classNames[i]} value={p} color={color} />)}
              </div>
            );
          })}
        </div>
      </details>
    </div>
  );
}

// Superpone la radiografía original y el mapa Grad-CAM; el control deslizante
// decide cuánto de cada una se ve (nativo <input type="range">: funciona con teclado y táctil).
function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50);
  return (
    <div>
      <div className="compare">
        <img src={after} alt="Mapa de calor Grad-CAM" />
        <img className="compare-top" src={before} alt="Radiografía original"
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <div className="compare-line" style={{ left: `calc(${pos}% - 1px)` }} />
        <span className="compare-tag" style={{ left: 8 }}>Original</span>
        <span className="compare-tag" style={{ right: 8 }}>Grad-CAM</span>
      </div>
      <input className="compare-range" type="range" min="0" max="100" value={pos}
        aria-label="Mover la línea para comparar la radiografía original (izquierda) con el Grad-CAM (derecha)"
        onChange={e => setPos(Number(e.target.value))} />
    </div>
  );
}

const DATASETS = [
  { name: 'Chest X-Ray Images (Pneumonia)', note: 'Kermany et al., niños de 1 a 5 años · CC BY 4.0',
    href: 'https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia' },
  { name: 'COVID-19 Radiography Database', note: 'Chowdhury, Rahman et al., IEEE Access 2020',
    href: 'https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database' },
  { name: 'Tuberculosis (TB) Chest X-ray Database', note: 'Rahman et al., IEEE Access 2020',
    href: 'https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset' },
];

// Transparencia sobre el origen de los datos y el alcance real del modelo
function DataSources() {
  return (
    <details style={{ marginTop: 20, borderTop: `1px solid ${T.border}`, paddingTop: 14 }}>
      <summary style={{ fontSize: 12, fontWeight: 600, color: T.textMid, cursor: 'pointer' }}>
        Fuentes de datos y limitaciones
      </summary>
      <div style={{ marginTop: 10, fontSize: 12, color: T.textMid, lineHeight: 1.6 }}>
        <p style={{ margin: '0 0 8px' }}>
          Los modelos se entrenaron con datasets públicos de Kaggle:
        </p>
        <ul style={{ margin: '0 0 10px', paddingLeft: 18 }}>
          {DATASETS.map(d => (
            <li key={d.name}>
              <a href={d.href} target="_blank" rel="noreferrer" style={{ color: T.primary }}>{d.name}</a>
              <span style={{ color: T.textDim }}> — {d.note}</span>
            </li>
          ))}
        </ul>
        <p style={{ margin: 0 }}>
          <strong>Limitaciones:</strong> como los datasets vienen de fuentes distintas, el modelo
          puede aprender rasgos propios de cada hospital o equipo (marcas, contraste, edad del paciente)
          en lugar de la enfermedad. Con radiografías de otros centros, o de pacientes distintos a los del
          entrenamiento, los resultados pueden ser poco fiables. Las métricas de COVID-19 y tuberculosis
          (AUC 1.000) son una señal de alerta, no de perfección. Más detalle en el README.
        </p>
      </div>
    </details>
  );
}

const SAMPLES = [1, 2, 3].map(n => `${process.env.PUBLIC_URL}/samples/ejemplo-${n}.jpg`);

function Analysis() {
  const [file,      setFile]      = useState(null);
  const [preview,   setPreview]   = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);
  const [activeTab, setActiveTab] = useState('gradcam');
  const [isDicom,   setIsDicom]   = useState(false);
  const [dragging,  setDragging]  = useState(false);

  const loadSample = async (src) => {
    try {
      const blob = await (await fetch(src)).blob();
      loadFile(new File([blob], src.split('/').pop(), { type: blob.type }));
    } catch { setError('No se pudo cargar la imagen de ejemplo'); }
  };

  const handleFileSelect = e => loadFile(e.target.files[0]);

  const loadFile = f => {
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) { setError('Máximo 10 MB'); return; }
    setFile(f); setError(null); setResult(null);
    const dicom = f.name.toLowerCase().endsWith('.dcm');
    setIsDicom(dicom);
    if (dicom) { setPreview(null); return; }
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const endpoint = isDicom ? '/predict/dicom' : '/predict/auto';
      const res = await authFetch(`${API_URL}${endpoint}`, { method: 'POST', body: form });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Error'); }
      setResult(await res.json());
      setActiveTab('gradcam');
    } catch (e) {
      setError(e instanceof TypeError ? 'No se pudo conectar con el servidor. Revisa que esté activo.' : e.message);
    }
    finally { setLoading(false); }
  };

  const previewSrc = preview || (isDicom && result?.gradcam?.heatmap_image) || null;

  return (
    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', fontFamily: T.font }}>
      {/* Upload panel */}
      <div className="upload-panel" style={{ background: T.surface, borderRadius: T.radiusLg, padding: 24, boxShadow: T.shadow, border: `1px solid ${T.border}` }}>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: T.text, margin: '0 0 6px', letterSpacing: '-0.01em' }}>
          Subir Radiografía
        </h2>
        <p style={{ color: T.textMid, fontSize: 13, margin: '0 0 20px', lineHeight: 1.6 }}>
          Análisis automático de neumonía, COVID-19 y tuberculosis. Soporta JPG, PNG y DICOM.
        </p>

        <div
          className={`dropzone${dragging ? ' is-dragging' : ''}`}
          role="button" tabIndex={0}
          aria-label="Seleccionar o soltar una radiografía"
          onClick={() => document.getElementById('fileInput').click()}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); document.getElementById('fileInput').click(); }
          }}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); loadFile(e.dataTransfer.files?.[0]); }}>
          <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.dcm" onChange={e => { handleFileSelect(e); e.target.value = ''; }} style={{ display: 'none' }} />
          <div style={{ width: 44, height: 44, borderRadius: 10, background: T.primaryLt, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
            <Upload size={20} color={T.primary} />
          </div>
          <div style={{ fontWeight: 600, fontSize: 14, color: T.text, marginBottom: 4 }}>
            {isDicom && file ? file.name : 'Seleccionar archivo'}
          </div>
          <div style={{ color: T.textDim, fontSize: 12 }}>
            {dragging ? 'Suelta la imagen aquí' : 'Arrastra una imagen o haz clic · JPG, PNG, DICOM · máx. 10 MB'}
          </div>
        </div>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: T.textDim, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            O prueba con un ejemplo
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {SAMPLES.map((s, i) => (
              <button key={s} onClick={() => loadSample(s)} disabled={loading}
                aria-label={`Cargar radiografía de ejemplo ${i + 1}`}
                style={{ padding: 0, border: `1px solid ${T.border}`, borderRadius: 8, overflow: 'hidden', cursor: 'pointer', flex: 1, background: T.bg }}>
                <img src={s} alt={`Ejemplo ${i + 1}`} style={{ width: '100%', height: 72, objectFit: 'cover', display: 'block' }} />
              </button>
            ))}
          </div>
        </div>

        {previewSrc && (
          <div style={{ textAlign: 'center', marginBottom: 16 }}>
            <img src={previewSrc} alt="Preview" style={{
              maxWidth: '100%', maxHeight: 160, borderRadius: 10,
              border: `1px solid ${T.border}`, boxShadow: T.shadow,
            }} />
            <div style={{ fontSize: 11, color: T.textDim, marginTop: 6 }}>{file?.name}</div>
          </div>
        )}

        {isDicom && result?.dicom_metadata && (
          <div style={card({ background: T.primaryLt, border: `1px solid #c7d7f9`, marginBottom: 16 })}>
            <div style={{ fontSize: 11, fontWeight: 700, color: T.primary, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 10 }}>
              Metadatos DICOM
            </div>
            {[['Paciente', result.dicom_metadata.patient_name], ['Fecha', result.dicom_metadata.study_date_formatted], ['Modalidad', result.dicom_metadata.modality], ['Institución', result.dicom_metadata.institution]].filter(([, v]) => v).map(([l, v]) => (
              <div key={l} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 12 }}>
                <span style={{ color: T.textMid, minWidth: 72 }}>{l}</span>
                <strong style={{ color: T.text }}>{v}</strong>
              </div>
            ))}
          </div>
        )}

        <button onClick={handleAnalyze} disabled={!file || loading} style={{
          ...btn('primary', { width: '100%', justifyContent: 'center', padding: '13px', fontSize: 14, borderRadius: 10 }),
          opacity: (!file || loading) ? 0.6 : 1,
        }}>
          {loading
            ? <><RefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} />{isDicom ? 'Procesando DICOM...' : 'Analizando...'}</>
            : <><Search size={15} />Analizar</>}
        </button>

        {error && (
          <div role="alert" style={{ marginTop: 12, padding: '10px 14px', borderRadius: 8, background: T.dangerLt, color: T.danger, fontSize: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertCircle size={13} />{error}
          </div>
        )}
      </div>

      {/* Results panel */}
      <div className="results-panel" style={{ background: T.surface, borderRadius: T.radiusLg, padding: 24, boxShadow: T.shadow, border: `1px solid ${T.border}` }}>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: T.text, margin: '0 0 20px', letterSpacing: '-0.01em' }}>Resultados</h2>
        {result ? (
          <div>
            {(() => {
              const topPos = result.ranking?.some(r => r.is_positive);
              const tone = topPos
                ? { color: T.danger, bg: T.dangerLt, border: '#fecaca' }
                : { color: T.success, bg: T.successLt, border: '#a7f3d0' };
              return (
                <div className="reveal" style={card({ background: tone.bg, border: `1px solid ${tone.border}`, borderLeft: `4px solid ${tone.color}`, marginBottom: 16 })}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: tone.color, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
                    Conclusión del Sistema
                  </div>
                  <p style={{ margin: 0, fontSize: 13, color: T.text, lineHeight: 1.6 }}>{prettyText(result.conclusion)}</p>
                </div>
              );
            })()}

            <div style={card({ background: T.warnLt, border: '1px solid #fde68a', marginBottom: 16, padding: '12px 16px' })}>
              <p style={{ margin: 0, fontSize: 12, color: '#78350f', lineHeight: 1.6, display: 'flex', gap: 8 }}>
                <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
                <span>Prototipo de investigación y portafolio. Los resultados no constituyen un diagnóstico médico y deben ser revisados por un profesional de la salud.</span>
              </p>
            </div>

            {result.ranking?.map((item, i) => <DiseaseCard key={item.disease} item={item} index={i} />)}

            {/* Visualizations */}
            <div style={card()}>
              <div style={{ fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 4 }}>Visualizaciones</div>
              <div style={{ fontSize: 11, color: T.textDim, marginBottom: 12 }}>
                {{
                  gradcam: 'Grad-CAM: zonas donde el modelo puso más atención. Rojo = mayor atención · azul = área normal.',
                  yolo: 'YOLO: cuadros sobre posibles lesiones localizadas en la radiografía.',
                  seg: 'U-Net: contorno de los pulmones, separado del resto de la imagen.',
                }[activeTab]}
              </div>
              <div role="tablist" style={{ display: 'flex', gap: 6, marginBottom: 14, flexWrap: 'wrap' }}>
                {[
                  { key: 'gradcam', label: 'Grad-CAM',  Icon: Eye },
                  { key: 'yolo',    label: `YOLO (${result.yolo?.total ?? 0})`, Icon: Search },
                  { key: 'seg',     label: 'Segmentación', Icon: Layers },
                ].map(t => (
                  <button key={t.key} role="tab" aria-selected={activeTab === t.key}
                    onClick={() => setActiveTab(t.key)} style={{
                    ...btn(activeTab === t.key ? 'primary' : 'ghost', { fontSize: 12, padding: '6px 12px' }),
                  }}>
                    <t.Icon size={12} />{t.label}
                  </button>
                ))}
              </div>

              {activeTab === 'gradcam' && (result.gradcam?.heatmap_image
                ? (preview && !isDicom
                    ? <CompareSlider before={preview} after={result.gradcam.heatmap_image} />
                    : <img src={result.gradcam.heatmap_image} alt="Grad-CAM" style={{ width: '100%', borderRadius: 8, border: `1px solid ${T.border}` }} />)
                : <div style={{ color: T.textDim, fontSize: 13, textAlign: 'center', padding: 20 }}>No disponible</div>)}

              {activeTab === 'yolo' && (
                <div>
                  {result.yolo?.image_url
                    ? <img src={result.yolo.image_url} alt="YOLO" style={{ width: '100%', borderRadius: 8, border: `1px solid ${T.border}`, marginBottom: 10 }} />
                    : <div style={{ color: T.textDim, fontSize: 13, textAlign: 'center', padding: 20 }}>No disponible</div>}
                  {result.yolo?.detections?.map((d, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 10px', borderRadius: 6, background: T.bg, marginBottom: 4, fontSize: 12 }}>
                      <span style={{ color: T.text }}>{d.class_name}</span>
                      <span style={{ fontWeight: 700, color: T.primary }}>{(d.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'seg' && (result.segmentation?.image_url
                ? <img src={result.segmentation.image_url} alt="Segmentación" style={{ width: '100%', borderRadius: 8, border: `1px solid ${T.border}` }} />
                : <div style={{ color: T.textDim, fontSize: 13, textAlign: 'center', padding: 20 }}>No disponible</div>)}
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: T.textDim }}>
            <div style={{ width: 64, height: 64, borderRadius: 16, background: T.primaryLt, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <Activity size={28} color={T.primary} />
            </div>
            <p style={{ fontSize: 15, color: T.textMid, margin: '0 0 16px', fontWeight: 500 }}>
              Sube una radiografía para comenzar
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
              {['Ensemble 3 modelos por enfermedad', 'Grad-CAM · YOLO · U-Net', 'Soporte DICOM nativo'].map(f => (
                <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: T.textDim }}>
                  <CheckCircle size={13} color={T.success} />{f}
                </div>
              ))}
            </div>
          </div>
        )}

        <DataSources />
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   MAIN APP
══════════════════════════════════════════════════════════════════════════════*/
export default function App() {
  const [loggedIn,     setLoggedIn]     = useState(!!localStorage.getItem('token'));
  const [page,         setPage]         = useState('analysis');
  const [user,         setUser]         = useState(null);
  const [pendingCount, setPendingCount] = useState(0);

  useEffect(() => {
    if (!loggedIn) return;
    authFetch(`${API_URL}/auth/me`).then(r => r.ok ? r.json() : null).then(data => { if (data) setUser(data); }).catch(() => {});
  }, [loggedIn]);

  useEffect(() => {
    if (!loggedIn) return;
    authFetch(`${API_URL}/studies`).then(r => r.json()).then(data => {
      const arr = extractStudies(data);
      setPendingCount(arr.filter(s => s.predictions?.[0] && !s.predictions[0].is_validated).length);
    }).catch(() => {});
  }, [loggedIn]);

  if (!loggedIn) return <Login onLogin={() => setLoggedIn(true)} />;
  const canValidate = user?.role === 'admin' || user?.role === 'radiologist';

  const PAGE_TITLES = {
    analysis:    'Análisis',
    dashboard:   'Dashboard',
    history:     'Historial de Estudios',
    patients:    'Pacientes',
    radiologist: 'Panel Radiólogo',
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', fontFamily: T.font, background: T.bg }}>
      <Sidebar active={page} onChange={setPage} onLogout={() => { localStorage.removeItem('token'); setLoggedIn(false); setUser(null); setPage('analysis'); }} user={user} pendingCount={pendingCount} />

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
        {/* Top bar */}
        <div style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: '14px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 1px 0 rgba(0,0,0,0.04)' }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 15, fontWeight: 700, color: T.text, letterSpacing: '-0.01em' }}>{PAGE_TITLES[page]}</h1>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {user && <RoleBadge role={user.role} />}
            <span style={{ fontSize: 11, color: T.textDim, fontFamily: T.mono }}>rt26 v2.0</span>
          </div>
        </div>

        <div className="page-pad" style={{ padding: 24, flex: 1 }}>
          {page === 'analysis'    && <Analysis />}
          {page === 'dashboard'   && <Dashboard />}
          {page === 'history'     && <History />}
          {page === 'patients'    && <Patients />}
          {page === 'radiologist' && canValidate && <RadiologistPanel />}
          {page === 'radiologist' && !canValidate && (
            <div style={{ ...card(), textAlign: 'center', padding: 80, color: T.textDim }}>
              <Shield size={40} style={{ marginBottom: 12, opacity: 0.3 }} />
              <p style={{ margin: 0 }}>Acceso restringido a radiólogos y administradores</p>
            </div>
          )}
        </div>
      </div>
      {/* El asistente de chat es opcional: requiere un servicio externo (puerto 8001) */}
      {process.env.REACT_APP_ENABLE_CHAT === 'true' && <ChatWidget />}
    </div>
  );
}