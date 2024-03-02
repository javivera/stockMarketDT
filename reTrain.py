from modelGenerator import generateModels
import csv
import sys

def reTrain(symbol:str,interval:str):
    ''' ad '''
    file_path = f'offline/{symbol}/{interval}/model-1/model-1-indicators.csv'
    # indicators = []
    indicators = set()
    try:
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            # Read the header row

            # Read the first and second rows
            first_row = next(csv_reader)
            second_row = next(csv_reader)
            third_row = next(csv_reader)
            forth_row = next(csv_reader)

            # Add elements from the first row to the list
            # indicators.extend(first_row)
            indicators.update(first_row)
            indicators.update(second_row) 
            indicators.update(third_row) 
            indicators.update(forth_row) 


    except FileNotFoundError:
        print("Error: CSV file not found.")
    except StopIteration:
        print("Error: CSV file does not have enough rows.")
    except Exception as e:
        print(f"Error: {e}")
    print(list(indicators))

    generateModels([f'{symbol}'],1,list(indicators))

arguments = sys.argv[1:]  # List of command-line arguments
reTrain(arguments[0],arguments[1])

