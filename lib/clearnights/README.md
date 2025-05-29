# ClearNights

A python module which implements the ClearNights algorithm developed by Dr. Ha Nguyen et. al.

# TODO - More details about usage

```python
from clearnights.clearnights import ClearNights

lst = load_your_lst_data()
latitude = ...
longitude = ...

clearnights = ClearNights()

output_df = clearnights.process_location(lst, None, longitude, latitude)
```