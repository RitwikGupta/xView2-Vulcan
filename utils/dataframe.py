import geopandas
import rasterio
import rasterio.warp
import rasterio.crs
from shapely.geometry import Polygon


def make_footprint_df(files):
    """
    Creates a dataframe of raster footprints and metadata
    :param files: iterable of roster filenames
    :return: geodataframe of raster footprints and metadata
    """
    # Todo: this requires much more error handling and tests

    polys = []
    filename = []
    res = []
    im_crs = []
    bounds = []
    height = []
    width = []


    # Get values from source rasters
    for f in files:
        with rasterio.open(f) as src:

            # Create footprint polygon
            vert = [
                (src.bounds.left, src.bounds.top),
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.right, src.bounds.bottom),
                (src.bounds.right, src.bounds.top)
            ]
            polys.append(Polygon(vert))

            filename.append(f)
            res.append(src.res)
            im_crs.append(src.crs)
            height.append(src.height)
            width.append(src.width)
            bounds.append(src.bounds)

    df = geopandas.GeoDataFrame(
        {
            'filename': filename,
            'crs': im_crs,
            'src_res': res,
            'height': height,
            'width': width,
            'bounds': bounds,
            'geometry': polys
        }
    )

    # Set CRS by CRS of first image
    df.crs = df['crs'][0]

    return df


def process_df(df, dest_crs):
    """
    Process geadataframe of raster footprints and adds tranform resolution
    :param df: geodataframe
    :param dest_crs: destination CRS to get transform resolution
    :return: geodataframe
    """
    df['trans_res'] = df.apply(lambda x: get_trans_res(x.crs, x.width, x.height, x.bounds, dest_crs), axis=1)
    # Remove rasters with resolution that wildly varies from the others
    # Remove rasters with CRS that is not the same as most?

    # Todo: handle our rasters with no geographic data
    # Remove rasters without transform resolution
    #df = df[df['trans_res'] is not None]

    # Remove rasters that do not have CRS set
    #df = df[df['crs'] is not None]

    return df


def get_utm(df):
    """
    Calculate UTM EPSG code for coordinate pair
        The EPSG is:
        32600+zone for positive latitudes
        32700+zone for negatives latitudes
    :param df: geodataframe
    :return: EPSG code
    """
    """Return UTM EPSG code of respective lon/lat.

    """

    cent = df.dissolve().geometry.to_crs(4326).centroid
    lon = cent.x[0]
    lat = cent.y[0]
    zone = int(round((183 + lon) / 6, 0))
    epsg = int(32700 - round((45 + lat) / 90, 0) * 100) + zone

    return rasterio.crs.CRS.from_string(f'EPSG:{epsg}')

    # Todo: This would be the preferred way, however it requires pyproj 3.0+ which seems to break every environment I tried.
    #return df.estimate_utm_crs()


def get_trans_res(src_crs, width, height, bounds, dst_crs):

    # Get our transform
    transform = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=width, height=height,
        left=bounds.left,
        bottom=bounds.bottom,
        right=bounds.right,
        top=bounds.top,
        dst_width=width, dst_height=height
    )

    return (transform[0][0], -transform[0][4])


def get_intersect(pre_df, post_df, args):

    """
    Computes intersect of input two rasters.
    :param pre_mosaic: pre mosaic
    :param post_mosaic: post mosaic
    :return: tuple of intersect in (left, bottom, right, top)
    """
    pre_env = pre_df.to_crs(args.destination_crs).unary_union
    post_env = post_df.to_crs(args.destination_crs).unary_union
    int = pre_env.intersection(post_env)
    assert int.area > 0

    return int.bounds


def get_max_res(pre_df, post_df):
    res_list = list(pre_df.trans_res) + list(post_df.trans_res)
    x = max(x[0] for x in res_list)
    y = max(x[1] for x in res_list)
    return (x, y)