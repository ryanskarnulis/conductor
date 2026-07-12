# syntax=docker/dockerfile:1
#
# Frontend image: build the Vite bundle, serve it with nginx which also
# reverse-proxies /api to the backend container. Build context is the repo root.

# --- build stage -----------------------------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Install with the lockfile first for a cached dependency layer. devDependencies
# are required — the build runs `tsc -b && vite build`.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Empty by default → the API client emits relative /api paths (see
# src/api/client.ts), which nginx serves same-origin and proxies to the backend.
# This is a compile-time constant, so it must be a build ARG, not a runtime env.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# --- serve stage -----------------------------------------------------------
FROM nginx:alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
