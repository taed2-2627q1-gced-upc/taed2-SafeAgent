# First presentation outline

This outline supports a presentation of about 15 minutes. The report draft and
linked project evidence provide the material for each part.

| Part | Time | Main point and evidence |
| --- | ---: | --- |
| Project goal | 1 minute | Classify command risk and explain the context question |
| Dataset | 2 minutes | Show the source, cleaned counts and fixed grouped split |
| Leakage controls | 2 minutes | Explain shared examples and excluded annotation fields |
| Models | 2 minutes | Present the baseline and paired encoder design |
| Pipeline | 2 minutes | Walk through prepare, checks, train and evaluate |
| Versioning and tracking | 2 minutes | Show GitHub, DVC recovery and the shared baseline run |
| First result | 2 minutes | Show macro F1, DENY recall and the false allow errors |
| Next work | 2 minutes | Explain pending encoder results, energy and further QA |

The demo should use existing outputs rather than training during the presentation.
Start from the source branch, show the data and model versions, then open the
shared run and its confusion matrix. Use the actual validation result and keep
pending encoder and energy results labelled as pending.

The team should agree speakers and contribution details before presenting.
Each speaker should connect the implementation choices with the project question
and be ready to explain how another teammate can reproduce the result.
