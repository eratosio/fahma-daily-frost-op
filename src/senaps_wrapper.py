import logging
import json
import os.path

from as_models.models import model
from as_models.runtime.python import Context

from eratos.creds import JobKeyCreds, AccessTokenCreds
from eratos.adapter import Adapter
from eratos.operator import Operator
from eratos.resource import Resource
from eratos.ern import Ern

logging.basicConfig(level=logging.INFO)


def load_json_doc(context: Context, id, reqValue: bool = True):
    doc = getattr(context.ports, id, None)
    if doc is None or doc.value is None or doc.value == "":
        if reqValue:
            raise KeyError(f"{id} is a required input port")
        else:
            return None
    try:
        return json.loads(doc.value)
    except:
        raise KeyError(f"{id} is not valid JSON")


@model("fahma.operators.daily.frost.metrics")
def eratos_operator_wrapper(context: Context):
    # Load the Eratos inputs.
    config = load_json_doc(context, "config")
    secrets = load_json_doc(context, "secrets")

    # Load the operator.
    if "tracker" in secrets and "jobKey" in secrets:
        creds = JobKeyCreds(key=secrets["jobKey"], tracker=secrets["tracker"])
    elif "id" in secrets and "secret" in secrets:
        creds = AccessTokenCreds(**secrets)
    else:
        raise Exception("Secrets do not contain valid credentials")

    adapter = Adapter(creds=creds)
    adapter._senaps_context = context

    # Load the operator.
    op = Operator(
        adapter, ern="ern:e-pn.io:resource:fahma.operators.daily.frost.metrics"
    )

    # Load the operator inputs.
    inputs = {}
    for inputDef in op.resource().prop("inputs"):
        k = inputDef["name"]
        inputs[k] = load_json_doc(
            context,
            "input_" + k,
            reqValue=inputDef["required"] if "required" in inputDef else False,
        )
        if inputs[k] is None:
            if "default" in inputDef:
                inputs[k] = inputDef["default"]
            else:
                del inputs[k]

    # Check the output document nodes exist.
    outDocNodes = {}
    for outputDef in op.resource().prop("outputs"):
        k = outputDef["name"]
        doc = getattr(context.ports, "output_" + k, "")
        if doc is None:
            raise KeyError(f"{k} is a required output port")
        outDocNodes[k] = doc

    # Run the operator.
    code_dir = os.path.dirname(__file__)
    outputs = op(code_dir, **inputs)

    # Output.
    for outputDef in op.resource().prop("outputs"):
        k = outputDef["name"]
        if k not in outputs:
            raise KeyError(f"{k} is missing from operator outputs")
        val = outputs[k]
        if outputDef["type"] == "resource":
            if type(val) is Resource:
                val = str(val.ern())
            elif type(val) is Ern:
                val = str(val)
            elif type(val) is str:
                try:
                    e = Ern(ern=val)
                    assert e.type() == "resource"
                except:
                    raise KeyError(
                        f"{k} output is an invalid ern or not of type resource"
                    )
            else:
                raise KeyError(f"{k} output has invalid type, expected resource")
        outDocNodes[k].value = json.dumps(val)