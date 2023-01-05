# Scale to convert the fontaine blue format into the V scale format, there is no official conversion and it shouldn't
# really matter for our purposes but this is one of the more widely used ones and can be found here:
# https://en.wikipedia.org/wiki/Grade_(bouldering)
font_scale_to_v_scale = {
    "1": "V2",
    "2": "V2",
    "3": "V2",
    "4": "V2",
    "4+": "V2",
    "5": "V2",
    "5+": "V2",
    "6A": "V2",
    "6A+": "V3",
    "6B": "V3",
    "6B+": "V4",
    "6C": "V5",
    "6C+": "V6",
    "7A": "V7",
    "7A+": "V7",
    "7B": "V8",
    "7B+": "V8",
    "7C": "V9",
    "7C+": "V10",
    "8A": "V11",
    "8A+": "V12",
    "8B": "V13",
    "8B+": "V14",
    "8C": "V15",
    "8C+": "V16",
    "9A": "V17",
}


def grade_converter(fb_grade):
    v_grade = font_scale_to_v_scale.get(fb_grade)

    # Just to make sure that we aren't inputting data into our models if the conversion failed, this shouldn't happen
    # but could be possible if the core data in the application changes
    if v_grade is None:
        raise Exception(f"V grade not found for {fb_grade}")

    return v_grade
