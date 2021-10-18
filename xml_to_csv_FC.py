import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def convert_class_to_int_label(label_text):
    conversion = {
        "Negative": 0,
        "Primordial": 1,
        "Primary": 2,
        "Secondary": 3,
        "Tertiary": 4,
    }
    return conversion[label_text]


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            label_text = member[0].text
            label_int = convert_class_to_int_label(label_text)

            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                label_text,
                label_int,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )

            xml_list.append(value)

    column_names = [
        "filename",
        "width",
        "height",
        "class",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    return xml_df


def main():
    # for folder in ["train", "test"]:
    xml_folder = os.path.abspath("./data/scenario1/Xml")
    parent_folder = os.path.dirname(xml_folder)
    output_filename = os.path.join(parent_folder, "labels.csv")

    # image_path = os.path.join(os.getcwd(), folder)
    print(f"Reading xml files in {xml_folder}")

    xml_df = xml_to_csv(xml_folder)
    xml_df.to_csv(output_filename, index=None)

    print("Successfully converted xml to csv: {output_filename}")


if __name__ == "__main__":
    main()
