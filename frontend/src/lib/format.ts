/** 465.4 -> "7m 45s" */
export function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** "2026-07-20T19:01:30+00:00" -> "2026-07-20" */
export function formatDate(iso: string) {
  return iso.slice(0, 10);
}

/** "EPIGENETIC_MODIFICATION" -> "Epigenetic modification" */
export function formatType(type: string) {
  const words = type.toLowerCase().replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** 2_411_724 -> "2.4 MB" */
export function formatBytes(bytes: number) {
  const mb = bytes / 1024 / 1024;
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
