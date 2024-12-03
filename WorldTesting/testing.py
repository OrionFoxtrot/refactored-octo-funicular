
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    map_name = sys.argv[1]
    
    env_map = Image.open(map_name)
    
    env_map = np.array(env_map)
    #plt.imshow(env_map)
    #plt.show()
    print(env_map.shape)

    debug_image = Image.fromarray(env_map)
    debug_image.save('debug_output.png')



if __name__ == '__main__':
    main()
