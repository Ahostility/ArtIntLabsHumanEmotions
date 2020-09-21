import os

def open_result_file(file_path):

    os.system("py -m detection_nlp_enotions.get_emotions_nlp " + file_path)

    with open("./text_out.txt", "r") as resultFile:
        for line in resultFile:
            answer = line.split("&")

    resultNLP = []
    middle = []

    for i in range(len(answer)):
        middle.append(answer[i][2:-2].split("'"))
        resultNLP.append((middle[i][0], int(answer[i][-2:-1])))

    return resultNLP
