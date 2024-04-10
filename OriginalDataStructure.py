class NGSet:
    def __init__(self, mark_txt, label_color_txt, data_list: list):
        self.mark_txt = mark_txt
        self.label_color_txt = label_color_txt
        self.data_list = data_list


class NGData:
    def __init__(self, bmp_image, bmp_image_t, xml_file):
        self.bmp_image = bmp_image
        self.bmp_image_t = bmp_image_t
        self.xml_file = xml_file
