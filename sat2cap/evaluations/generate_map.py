import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def generate_map(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)

    # Create basemap object
    plt.figure(figsize=[25, 10])
    map = Basemap()

    # Draw map features
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color='gray')

    # Plot points
    x, y = map(data['long'].values, data['lat'].values)
    #map.scatter(x, y, c=data[['red','green','blue']].values)
    map.scatter(x, y)

    # Add latitude and longitude ticks
    map.drawparallels(range(-90, 90, 10), labels=[1, 0, 0, 0])
    map.drawmeridians(range(-180, 180, 10), labels=[0, 0, 0, 1])

    # Show and save plot
    plt.show()
    plt.savefig(output_path)

if __name__ == '__main__':
    print('In')
    input_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/pca_visualizations/pca_results.csv'
    output_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/pca_visualizations/data.jpeg'
    generate_map(input_path, output_path)