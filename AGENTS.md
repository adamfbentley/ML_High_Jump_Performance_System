# AGENTS.md

## Sources Of Truth

The technical source of truth for the codebase is .github/copilot-instructions.md.
It defines the project goal, architecture, physics conventions, datasets, data structures,
and code conventions.

The file questions to imogen.md contains the questions asked from the athlete perspective.
The folder imogens_response- to-questions/ contains Imogen's answers to those questions.
Use that folder as the authoritative domain and user-priority reference for what the ideal
system should measure, prioritize, and report from a high-jumping perspective.

Do not treat imogens_response- to-questions/ as the primary technical specification for
architecture, naming, file structure, or implementation conventions. Use it to guide domain
priorities, desired metrics, and coach-usable outputs.

If there is a conflict between technical implementation guidance and the athlete reference,
follow .github/copilot-instructions.md for technical conventions and flag the conflict.

## Required Workflow For All Agents

1. Read .github/copilot-instructions.md first for technical context.
2. Read the relevant sections of imogens_response- to-questions/Highjumpproject.html when the task touches movement analysis, reporting, athlete priorities, or desired outputs.
3. Inspect the related source files and tests.
4. Identify concrete mismatches between the implementation, the technical conventions, and the athlete-domain priorities.
5. Apply precise fixes at the root cause.
6. Add or update tests when behavior changes.
7. Re-run the relevant commands or tests.

## Domain Priorities From Imogen

- Organize analysis around approach, curve, penultimate, takeoff, and flight.
- Prioritize takeoff speed, takeoff-foot ground contact time, body angle, straight-line body alignment, foot-to-mat angle, arm-drive timing, knee-drive timing, stride rhythm, all-stride contact quality, curve start, and curve adherence.
- Treat box drop jump, especially single-leg drop jump, as more relevant transfer work than CMJ.
- Produce coach-usable deviation outputs such as: "5th stride too short", "third stride stepped off the curve", or "foot contact time on penultimate stride was too long."

## Project Conventions

- Preserve the physics, units, and data-model conventions documented in .github/copilot-instructions.md.
- Use BiomechanicalSample as the canonical cross-dataset format.
- Do not introduce mock data in place of real physics or real datasets.
- Do not weaken physics-law checks to hide implementation issues.
