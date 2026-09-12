"""BiomedCAT local web console: a job queue over the pipeline and a Cytoscape viewer.

    uv run python -m biomedcat.webapp          # starts http://127.0.0.1:8765 and opens the browser

Design
------
* One FastAPI process serves a static single-page interface (no Node.js, no build step) and a
  small JSON API. Everything stays on the machine: the server binds to localhost only.
* A job is a set of documents dropped by the user plus the profiles to run: the default profiles
  of data/profiles and any custom profile built in the console (one priority 0 / 0.5 / 1 per
  entity branch of the Biolink model and a directionality setting; the class scope and the
  predicate weights are derived on the fly by `biomedcat.weights` and written into the job
  directory for the trace). Jobs are queued and
  executed one at a time by a worker thread, each in its own subprocess (`biomedcat.webapp.runner`),
  so a crash or a cancellation never takes the console down and the models of two jobs never
  coexist in memory.
* Each job owns a directory under `output/jobs/<job id>/`: job.json (state, progress, timings),
  job.log (full trace), the per-document results, and one context graph per profile with its
  Cytoscape export. The zip download is that directory plus a generated trace README.
* The viewer reads the Cytoscape JSON built by `biomedcat.webapp.graphs`: nodes grouped into
  clusters derived from the Biolink hierarchy, coloured by Biolink category, with the profile
  scope, provenance (seeds, slides) and, when the KG2c extras tables exist, descriptions and
  publications from RTX-KG2c.
"""
