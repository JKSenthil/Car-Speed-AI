def frame2speed_array(fpath):
    """
    Returns array, where index=frame and value=speed
    """
    file = open(fpath)
    frame2speed = file.readlines()
    # Clean up data, convert string to float 
    for i in range(len(frame2speed)):
        frame2speed[i] = float(frame2speed[i].strip())
    return frame2speed

# TODO need to implement equalize_adapthist next
