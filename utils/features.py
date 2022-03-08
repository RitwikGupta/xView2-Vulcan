import rasterio
from rasterio.features import dataset_features
from osgeo import gdal
import geopandas


def create_polys(in_files, threshold=30):

    """
    Create palygons to use for feature creation.
    :param in_files: List of DMG files to create polygons from.
    :return: Shapely polygons.
    """

    polygons = []
    for idx, f in enumerate(in_files):
        src = rasterio.open(f)
        crs = src.crs
        transform = src.transform

        bnd = src.read(1)
        polygons += list(dataset_features(src, 1, geographic=False))

    # Create geo dataframe
    df = geopandas.GeoDataFrame.from_features(polygons, crs=crs)
    df.rename(columns={'val': 'dmg'}, inplace=True)

    # Fix geometry if not valid
    df.loc[~df.geometry.is_valid, 'geometry'] = df[~df.geometry.is_valid].geometry.apply(lambda x: x.buffer(0))

    # Drop damage of 0 (no building), dissolve by each damage level, and explode them back to single polygons
    df = df.dissolve(by='dmg').reset_index().drop(index=0)
    df = df.explode().reset_index(drop=True)

    # Apply our threshold
    df['area'] = df.geometry.area
    df = df[df.area >= threshold]

    return df.reset_index(drop=True)


def write_output(features, out_file, layer='features'):
    features.to_file(out_file, driver='GPKG', layer=layer)

    return out_file


def create_aoi_poly(features):

    """
    Create convex hull polygon encompassing damage polygons.
    :param features: Polygons to create hull around.
    :return: ARCGIS polygon.
    """
    hull = features.dissolve().convex_hull
    df = geopandas.GeoDataFrame.from_features(hull, crs=features.crs)
    return df


def create_centroids(features):

    """
    Create centroids from polygon features.
    :param features: Polygon features to create centroids from.
    :return: List of ARCGIS point features.
    """

    cent_df = geopandas.GeoDataFrame.from_features(features.centroid, crs=features.crs)
    cent_df['dmg'] = features.dmg
    return cent_df


def rasterize(in_feats, out_file, out_shape):
    df = geopandas.read_file(in_feats)

    transform = rasterio.transform.from_bounds(*df['geometry'].total_bounds, *out_shape)
    shapes = ((shape) for shape in df.geometry)


    image = rasterio.features.rasterize(
            shapes,
            out_shape=out_shape,
            all_touched=True,
            transform=transform,
            )

    assert image.sum() > 0

    with rasterio.open(
            out_file, 'w',
            driver='GTiff',
            dtype=rasterio.uint8,
            count=1,
            width=out_shape[0],
            height=out_shape[1]) as dst:
        dst.write(image, indexes=1)

    return image