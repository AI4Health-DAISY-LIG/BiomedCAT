"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, AlertCircle, PlayCircle, Loader2 } from "lucide-react";

/**
 * Type definition for the notification state to avoid using 'any'.
 */
type NotificationType = "success"  | "error" | "info" | "loading";

interface PipelineNotification {
  message: string;
  type: NotificationType;
}

/**
 * A Client Component that listens to the SSE stream from the FastAPI bridge.
 * It displays transient notifications (toasts) when pipeline events occur.
 */
export default function PipelineEventListener() {
  const [notification, setNotification] = useState<PipelineNotification | null>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const eventSource = new EventSource(`${apiUrl}/events`);

    const showToast = (message: string, type: Notification 
      NotificationType) => {
      setNotification({ message, type });
      setIsVisible(true);
      // Auto-hide after 5 seconds
      const timeoutId = setTimeout(() => setIsVisible(false), 5000);
      return timeoutId;
    };

    // We use a ref or a local variable to track the active timeout if needed, 
    // but for simple toasts, a standard timeout is sufficient.

    eventSource.addEventListener("on_start", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      // Using file_id as defined in backend events.py
      showToast(`Started processing: ${data.file_id}`, "info");
    });

    eventSource.addEventListener("on_stage_change", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      // Using file_id, stage, and status as defined in backend events.py
      showToast(`${data.stage}: ${data.status}...`, "loading");
    });

    eventSource.addEventListener("on_success", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      // Using file_id as defined in backend events.py
      showToast(`Finished: ${data.file_id}`, "success");
    });

    eventSource.addEventListener("on_error", (e) => {
      const data = JSON.parse(e.target instanceof MessageEvent ? e.target.data : "{}");
      // Using file_id and error as defined in backend events.py
      showToast(`Error in ${data.file_id}: ${data.error}`, "error");
    });

    return () => eventSource.close();
  }, []);

  if (!isVisible || !notification) return null;

  const icons = {
    success: <CheckCircle2 className="text-green-5 0" />,
    error: <AlertCircle className="text-red-500" />,
    info: <PlayCircle className="text-blue-500" />,
    loading: <Loader2 className="animate-spin text-accent" />,
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-in fade-in slide-in-from-bottom-4">
      <div className="flex items-center gap-3 rounded-xl border border-line bg-card p-4 shadow-lg">
        {icons[notification.type]}
        <p className="text-sm font-medium text-body">{notification.message}</p>
      </div>
    </div>
  );
}
