import exifread


def find_location(image_path):
    with open(image_path, 'rb') as f:
        exif_data = exifread.process_file(f)

    latitude = exif_data.get('GPS GPSLatitude')
    latitude_ref = exif_data.get('GPS GPSLatitudeRef')
    longitude = exif_data.get('GPS GPSLongitude')
    longitude_ref = exif_data.get('GPS GPSLongitudeRef')

    if latitude and longitude:
        if latitude_ref.values[0] is 'S':
            latitude = -latitude
        if longitude_ref.values[0] is 'W':
            longitude = -longitude

        return latitude, longitude

    return None
