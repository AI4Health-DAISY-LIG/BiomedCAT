"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, AlertCircle, PlayCircle, Loader2 } from "lucide-react";

/**
 * A Client Component that listens to the SSE stream from the FastAPI bridge.
 * It displays transient notifications (toasts) when pipeline events occur.
 */
export default function PipelineEventListener() {
  const [notification, setNotification] = useState<any>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const eventSource = new EventSource(`${apiUrl}/events`);

    const showToast = (message: string, type: "success" | "error" | "info" | "loading") => {
      setNotification({ message, type });
      setIsVisible(true);
      // Auto-hide after 5 seconds
      setTimeout(() => setIsVisible(false), 5000);
    };

    eventSource.addEventListener("on_start", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      showToast(`Started processing: ${data.file_id}`, "info");
    });

    eventSource.addEventListener("on_stage_change", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      showToast(`${data.stage}: ${data.status}...`, "loading");
    });

    eventSource.addEventListener("on_success", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      showToast(`Finished: ${data.file_id}`, "success");
    });

    eventSource.addEventListener("on_error", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      showToast(`Error in ${data.file_id}: ${data.error}`, "error");
    });

    return () => eventSource.close();
  }, []);

  if (!isVisible || !notification) return null;

  const icons = {
    success: <CheckCircle2 className="text-green-500" />,
    error: <AlertCircle className: "text-red-500" />,
    info: <PlayCircle className="text-blue-500" />,
    loading: <Loader2 className="animate-spin text-accent" />,
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-in fade-in slide-in-from-bottom-4">
      <div className="flex items-center gap-3 rounded-xl border border-line bg-card p-4 shadow-lg">
        {icons[notification.type as keyof typeof icons]}
        <p className="text-sm font-medium text-body">{notification.message}</p>
      </div>
    </div>
  );
}
