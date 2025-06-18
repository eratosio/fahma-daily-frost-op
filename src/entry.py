from eratos.resource import Resource
from eratos.operator import operator
from daily_frost_metrics_on_demand import daily_frost_metrics
from typing import Union

@operator('ern:e-pn.io:resource:fahma.operators.daily.frost.metrics')
def entry(
        context,
        start_date: str,
        geom : str,
        end_date: str,
        frost_threshold: float,
        duration_threshold: float,
    ):
    eadapter = context['adapter']
    ecreds = eadapter._tracker_exchange._creds

    return daily_frost_metrics(
        geom,
        start_date,
        end_date,
        frost_threshold,
        duration_threshold,
        ecreds
    )




