# IMPORTS - START
import cv2
import numpy as np
import time
# IMPORTS - END

# CONSTANTS (FROM camera_controller.py) - START
GREEN_LOWER_BOUND = np.array([48, 40, 100])
GREEN_UPPER_BOUND = np.array([55, 255, 255])
RED_LOWER_BOUND = np.array([174, 200, 100])
RED_UPPER_BOUND = np.array([180, 255, 200])
# CONSTANTS (FROM camera_controller.py) - END

# HELPER FUNCTIONS (NO NEED TO COPY) - START
def get_masks(image): # filter the frame for red and green
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_green = np.all((hsv_image >= GREEN_LOWER_BOUND) & (hsv_image <= GREEN_UPPER_BOUND), axis=-1)
        mask_red = np.all((hsv_image >= RED_LOWER_BOUND) & (hsv_image <= RED_UPPER_BOUND), axis=-1)
        return mask_green, mask_red

def show_img_path(path, title='image'): # display an image
    image = cv2.imread(path)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    return image

def show_img(image, title='image'): # display a frame
    cv2.imshow(title, image)
    cv2.waitKey(0)

def convert_mask2img(mask): # convert a mask to a drawable frame
    return mask.astype(np.uint8)*255

def create_largest_component_mask(mask, pixel_list): # display a pixel list using a certain frame (only for size)
    height = mask.shape[0]
    width = mask.shape[1]
    new_mask = np.zeros((height, width), dtype=bool)
    for i in pixel_list:
        new_mask[i[0] - 1, i[1] - 1] = True
    return new_mask
# HELPER FUNCTIONS (NO NEED TO COPY) - END

### COPY THIS - START
def get_first_true_pixel_coordinates(mask): # function gets seed coordinates to find the first connected component
    first = np.argmax(mask.flatten()) # flatten the entire masked frame into a 1d array and find the first "white" (true) pixel
    filled_rows = int((first + 1) / mask.shape[1]) # convert position in 1d array to 2d coordinates, top to bottom filled rows where the entire row is "black" (false)
    row = 1 + filled_rows if (first + 1) % mask.shape[1] != 0 else filled_rows # last column gives int value in division, therefore we must +1 for other columns in the row index
    col = 1 + first - ((row - 1) * mask.shape[1]) # get column
    return (row, col) # return coordinates

def check_adjacents(mask, current_coordinates): # check adjacent pixels to the right and below whether they are true
    right = True if mask[current_coordinates[0] - 1, current_coordinates[1]] else False # get boolean of pixel to the right
    down = True if mask[current_coordinates[0], current_coordinates[1] - 1] else False # get boolean of pixel below
    boolean = right | down # check if either of the two is true (else, the connected component ends here approximately since we are dealing with convex shapes)
    return boolean, right # return values

def get_leftmost(mask, current_coordinates): # get the leftmost "white" (true) pixel after moving down one row
    while mask[current_coordinates[0] - 1, current_coordinates[1] - 2]: # as long as the pixel to the left is "white" (true)...
        current_coordinates = (current_coordinates[0], current_coordinates[1] - 1) # ...update the current coordinates
    return current_coordinates # return the leftmost "white" (true) coordinates of the current row within the component

def get_next_coordinates(mask, current_coordinates): # decide which is the next coordinate to evaluate
    boolean, right = check_adjacents(mask, current_coordinates) # get values from the adjacent check
    if not boolean: # if neither the pixel to the right or below is "white" (true)...
        return None, None # ... return empty values
    elif right: # if the pixel to the right is "white" (true)...
        row = current_coordinates[0]
        col = current_coordinates[1] + 1
        return True, (row, col) # ... go to pixel to the right
    else: # if there is an adjacent "white" (true) pixel but it is not to the right, then it must be in the row below
        row = current_coordinates[0] + 1
        col = current_coordinates[1]
        return False, (row, col)

def get_components(mask, debug=False): # find all the connected components (approximately)
    current_mask = mask.copy() # we do not want to operate on the actual mask that we feed into this function but create a copy of it
    seed_coordinates = get_first_true_pixel_coordinates(current_mask) # get seed coordinates
    components = [] # create empty list of all components that we will find, each element of this list will contain all pixel coordinates that are part of its component
    current_component = [seed_coordinates] # create a working list for the first component initially containing only the seed coordinates
    current_coordinates = seed_coordinates # set the working coordinates to be the seed coordinates initially
    while np.sum(np.where(current_mask, current_mask, current_mask)) > 0: # Continue finding components as long as the frame contains "white" (true) pixels
        while current_coordinates != None: # In the current component, continue as long as there are neighboring "white" (true) pixels
            go_right, next_coordinates = get_next_coordinates(current_mask, current_coordinates) # get next coordinates and whether it is on the right or not
            if (not go_right) and (go_right != None): # if the next coordinate exists and is not below ...
                next_coordinates = get_leftmost(current_mask, next_coordinates) # ... then we are in the next row, so find the leftmost "white"(true) pixel in this row
            if next_coordinates != None: # Put the current coordinates into the current component that is being discovered unless it is fully discovered (None)
                current_component.append(next_coordinates) 
            current_coordinates = next_coordinates # Update working coordinates
        for i in current_component: # after current component has been fully discovered, remove each of it pixels from the frame by setting it to "black" (false)
            current_mask[i[0] - 1, i[1] - 1] = False
        components.append(current_component) # put the list of pixels of the current component into the overall list of components
        seed_coordinates = get_first_true_pixel_coordinates(current_mask) # from the updated frame (current component removed from ti), find the next seed coordinates
        current_component = [seed_coordinates] # initialize the list of the new current component 
        current_coordinates = seed_coordinates # initialize the working coordinates of the new current component
        if debug: # debug option allows to show the updated frame after removal of each component
            show_img(convert_mask2img(current_mask))
    if debug: # debug option allows to show the empty frame after all components have been found
        show_img(convert_mask2img(current_mask))
    return components # return the list of components each containing a list of pixels

def get_largest_component(components): # from the list of components, find the one with the most pixels (largest area)
    components_size = [len(i) for i in components]
    largest_component_index = np.argmax(components_size)
    largest_pixel_list = components[largest_component_index]
    return components_size[largest_component_index], largest_pixel_list # return size of largest component and the list of pixels
### COPY THIS - END





# Below here is a test run on the sample image

path = '/mnt/hgfs/Shared Folder/sample.jpeg' # define sample image filepath
image = cv2.imread(path) # read image into cv2

green, red = get_masks(image) # convert image to masks
green_img, red_img = convert_mask2img(green), convert_mask2img(red) # create displayable image for debugging

iterlist_green = [] # helper list for the green mask
iterlist_red = [] # helper list for the red mask
n = 1000 # number of iterations
start_time = time.time() # get start time of execution
for i in range(n):
    iterlist_green.append(get_largest_component(get_components(green))) # get list of approximate components of green mask
    iterlist_red.append(get_largest_component(get_components(red))) # get list of approximate components of red mask
end_time = time.time() # get end time of execution
run_time = end_time - start_time
print('Total run time of ', n, ' executions [s]: ', run_time)
print('Average run time for one red-green mask pair [s]: ', run_time / 1000)
# expected values: ttl ~ 11s, avg ~ 0.01s

# To display the findings...
size_green, pixels_green = get_largest_component(get_components(green))
size_red, pixels_red = get_largest_component(get_components(red))
new_mask_green = create_largest_component_mask(green, pixels_green)
new_mask_red = create_largest_component_mask(red, pixels_red)
print('Approximate size of largest connected green component [px]: ', size_green)
print('Approximate size of largest connected red component [px]: ', size_red)

show_img(convert_mask2img(green), title='green_mask')
show_img(convert_mask2img(new_mask_green), title='green_mask_new')
show_img(convert_mask2img(red), title='red_mask')
show_img(convert_mask2img(new_mask_red), title='red_mask_new')

# close the windows
cv2.destroyAllWindows()