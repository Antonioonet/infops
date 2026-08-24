# AGENTS.md

## Communication and teaching style

- For technical questions, answer so the user can understand and independently reason about the subject—not merely copy commands.
- Start with the practical outcome, then explain the mental model and why each step is needed.
- Define unfamiliar terms in plain language and distinguish facts, assumptions, and recommendations.
- Give commands that are ready to adapt, identify placeholders explicitly, and explain what successful output should look like.
- Include a short verification step and the most likely failure modes with diagnostic checks.
- Prefer a complete, self-contained answer over a terse command dump. Avoid unnecessary jargon and unexplained acronyms.
- When there are multiple valid approaches, recommend one and briefly explain the trade-offs.
- Do not assume that a command was successful or that the user understood a concept; make the expected result observable.
- Ask a clarifying question only when the missing information materially changes the safe or correct solution. Otherwise make a clearly stated reasonable assumption.

## Academic experiment code

- Treat benchmark code as reproducible academic experiment code.
- When adapting an existing experiment, preserve its structure, control flow, naming style, metrics, output files, and checkpoint behavior as closely as possible; prefer a direct copy-and-adapt over introducing abstractions or framework-style validation.
- Keep the implementation clean and compact. Add only checks required for the experiment to run correctly or to prevent silent result corruption.
- Record the seed, dataset, model hyperparameters, runtime, metrics, and checkpoint-selection criterion in the experiment outputs.

## HPC and Jupyter defaults

- Treat the HPC compute node and the user's browser as separate machines. The Jupyter process runs on the HPC node; an SSH local port-forward connects the browser to it.
- Prefer starting Jupyter on a compute allocation, not on a login node, unless the site's policy explicitly permits login-node workloads.
- Keep Jupyter authentication enabled. Do not recommend disabling token/password authentication or exposing the Jupyter port directly to the public internet.
- Prefer an SSH tunnel (`ssh -N -L LOCAL_PORT:127.0.0.1:REMOTE_PORT ...`) so the Jupyter port remains private to the HPC node.
- Use a random or explicitly chosen high port, check that it is free, and explain how to map the local and remote ports.
- Account for schedulers such as Slurm or PBS: provide an interactive allocation example only when the scheduler is known; otherwise mark scheduler commands as templates.
- Explain how to stop both the Jupyter process and the SSH tunnel, and warn that closing the SSH session may terminate a tunnel or job depending on how it was launched.
- Never put passwords, tokens, private keys, or other secrets into repository files or shell history.

## Technical answer checklist

1. State the recommended architecture and why it is appropriate.
2. List prerequisites and identify site-specific values.
3. Provide the smallest working procedure.
4. Explain the network path and authentication model.
5. Verify each important step.
6. Provide targeted troubleshooting commands and likely causes.
7. Mention security, resource, and cleanup considerations.
