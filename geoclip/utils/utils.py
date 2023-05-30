#give lat long from path
def path_to_lat(path):
    img_name = path.split('/')[-1]
    splits = img_name.split('_')
    lat = splits[2]
    long = splits[3].replace('.jpg', "")
    return float(lat), float(long)

