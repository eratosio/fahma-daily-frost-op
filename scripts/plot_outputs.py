import os
import matplotlib.pyplot as plt
from eratos.creds import AccessTokenCreds

from dotenv import load_dotenv
load_dotenv()


def check_outputs():
    # Use the test_local_run_like_operator to get new ERNs
    min_temp_ern, frost_hours_ern, duration_ern = "ern:e-pn.io:resource:GGW4Q7NQXXHJ7PMV2NHY7Q4I", "ern:e-pn.io:resource:L46I3MN7EKBTR6ZQ3UDVUK66", "ern:e-pn.io:resource:THVITD6DT2U33BSIJOQTFY34"

    creds = AccessTokenCreds(os.environ.get("ERATOS_ID"), os.environ.get("ERATOS_SECRET"))

    get_output(frost_hours_ern, 'frost_hours', creds)
    get_output(min_temp_ern, 'min_temp', creds)
    get_output(duration_ern, 'duration', creds)


def get_output(ern, variable, creds):
    import xarray as xr
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    ds = xr.open_dataset(ern, eratos_auth=creds)
    ds.to_netcdf(f'{variable}.nc')

    # Select first timestep for visualization
    first_time = ds.time.values[0] if len(ds.time) > 0 else None
    if first_time is not None:
        ds_single = ds.sel(time=first_time)
        temp_data = ds_single[variable].values

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        # Add coastlines and borders
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # Create temperature contour plot
        lon = ds_single.lon.values
        lat = ds_single.lat.values
        im = ax.contourf(lon, lat, temp_data, levels=20,
                         transform=ccrs.PlateCarree(),
                         cmap='coolwarm')

        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.5)
        ax.xformatter = LONGITUDE_FORMATTER
        ax.yformatter = LATITUDE_FORMATTER

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label(variable)

        # Set title
        ax.set_title(f'{variable} - {str(first_time)[:10]}')

        plt.tight_layout()
        plt.savefig(f'{variable}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Map saved as '{variable}.png' for timestep: {first_time}")
    else:
        print("No time data available in dataset")

    pass
if __name__=='__main__':
    check_outputs()