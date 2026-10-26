# ─────────────────────────────────────────────
#  STAGE 1: builder — compila React
# ─────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copiar package files primero (mejor cache)
COPY package.json package-lock.json* ./

# Instalar dependencias
RUN npm ci

# Copiar código fuente
COPY . .

# Variables de entorno para el build
# La URL del backend en producción (Nginx la proxea internamente)
ARG REACT_APP_API_URL=/api
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Compilar para producción
RUN npm run build

# ─────────────────────────────────────────────
#  STAGE 2: runtime — Nginx sirve el build
# ─────────────────────────────────────────────
FROM nginx:1.25-alpine AS runtime

# Remover config default de Nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copiar nuestra config de Nginx
COPY nginx.conf /etc/nginx/conf.d/app.conf

# Copiar el build de React
COPY --from=builder /app/build /usr/share/nginx/html

# Puerto que expone Nginx (el docker-compose lo mapea)
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]