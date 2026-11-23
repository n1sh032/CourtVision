# Architecture and development

UI: `viewer/` owns video presentation and review labels. Transport: `pipeline/src/courtvision/server.py` owns same-origin local requests and bounded uploads. Application: `application.py` coordinates inference, progress and event filtering. Domain behavior: `track.py`, `contact.py`, and `contact_model.py` own tracking, event geometry and learned filtering respectively. The CLI also calls these domain services.

Run `courtvision serve` from the root and open `http://courtvision.localhost:8000`. The numeric loopback address still works for existing scripts. This branch retains the detector and label formats, with a separate inference service that can be tested without running YOLO. The single-clip pilot remains an experiment requiring independent evaluation.

## Engineering conventions

Each repository is independently runnable. Presentation calls a transport adapter, which invokes application behavior. I/O stays at explicit boundaries so tests can replace external providers. Long browser operations have one owner, reject overlapping requests, and report success or failure through the UI. Relay uses async provider requests; video and desktop work use a worker or an existing sequential CLI loop.

New behavior should validate input before touching a user's files or desktop, report partial failures accurately, and have a regression test for a concrete failure mode. Run the repository's checks before committing and document the exact manual checks separately from mocked tests.

The local apps share a request/status pattern, accessible form controls, and a responsive layout. Small standard-library transports are kept inside each repository so they do not require a sixth repository at runtime. The matching transport modules should be updated together when changing that contract.

## Local addresses

The browser resolves `.localhost` names to the loopback interface. These addresses are scoped to this computer. Start the Python process before opening the address. Chrome and Edge support this setup. A distinct port lets each project run alongside the others without an OS hosts-file change.

## Evaluation

Automated checks verify the workflows represented by their fixtures. Real model access, physical microphones and desktop focus still need environment-specific checks. CI is included under `.github/workflows/checks.yml`; its remote result must be checked after pushing.

Remaining work should be driven by observations from real use. Dataset accuracy, provider access and desktop-device behavior cannot be established by a successful test run alone.

## Verification for this change

112 pipeline tests passed and Ruff passed. The existing real-inference smoke workflow was rechecked through the branded local address. Detection accuracy still requires labeled independent clips.
