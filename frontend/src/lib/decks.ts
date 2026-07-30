// Facts about input decks, shared by the server and the browser — no node imports.

/** Mirrors SUPPORTED in biomedcat/pipeline.py. */
export const SUPPORTED = [".pptx", ".pdf", ".png", ".jpg", ".jpeg"];

/** A deck in Dataset/ that the pipeline has not processed yet. */
export type QueuedDeck = {
  name: string;
  bytes: number;
};
