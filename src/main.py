


def get_camera_params():
    result = []
    with open('camera_params.csv', 'r') as f:
        for line in f:
            nums = line.split(',')
            for n in nums:
                result.append(float(n))
            break
    return result



