import cv2
from plate_reader import PlateReader


def test(path, plate_reader):
    img = cv2.imread(path)
    parking_spot, text, _ = plate_reader.process_image(img)
    if (parking_spot is not None):
        print("Parking: {}, {}".format(parking_spot, text))


def main():
    plate_reader = PlateReader('assets/num_model.h5', 'assets/char_model.h5')
    for i in range(1):
        print("image {}".format(i))
        test('image{}.png'.format(i), plate_reader)


if __name__ == "__main__":
    main()
