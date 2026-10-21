# SafeAgent live demo

Duration: about two minutes. The requests go to the real Virtech deployment.

Verified on 21 October 2026 using the Swagger buttons: all the requests below
worked, and the automatic check passed 7/7. The loaded model was `vm_linear_svm`.

Open **http://nattech.fib.upc.edu:40490/docs**.

1. Expand **GET /ready**, click **Try it out** and **Execute**. Show HTTP 200,
   `status: ready`, the loaded model name and `model_version`.
2. Expand **POST /predict** and click **Try it out**. Use the **Examples** selector
   to send the following requests, clicking **Execute** after each selection.

| Example | Request | Verified result |
| --- | --- | --- |
| Classify git status | `{"command":"git status"}` | HTTP 200 and `ALLOW` |
| Classify a destructive command without executing it | `{"command":"rm -rf /"}` | HTTP 200 and `DENY` |
| Show input validation with an empty command | `{"command":""}` | HTTP 422 with validation details |

The successful responses include the same model version shown by `/ready`.
The service only classifies the submitted text.

Suggested explanation:

> First, the readiness check shows that the model is loaded. Then we send two
> commands and get different predictions. Finally, an empty command is rejected.
> These are requests to the deployed service, and the commands are never executed.

The complete quick check can also be run from the repository:

```sh
python scripts/smoke_test.py --demo --url http://nattech.fib.upc.edu:40490
```

It checks readiness, health, Swagger, the OpenAPI schema, both predictions and
invalid input. Success prints **7/7 checks passed**. A failed check returns a
nonzero exit code.
