def main():
    input_file = open("../mknap2.txt", "r")
    file_read = input_file.read()
    splitted_input = file_read.split("+++++++++++++++++++++++++++++")
    # Starting from index 4
    splitted_input = splitted_input[4:]
    for i in range (0, len(splitted_input), 2):
        filename_string = splitted_input[i]
        filename = filename_string.split()[1]
        input_data = splitted_input[i + 1]

        data_file = open(f"./{filename}.txt", "w")
        data_file.write(input_data.strip("\n"))
if __name__ == "__main__":
    main()