"""
//******************************************************************************
// FILENAME:           dove_vtext.py
// DESCRIPTION:        This Natural Language Processing Python script contains pseudocode for the VTE symptom extractor
//                     VTExt developed for the DOVE Project.
// AUTHOR(s):          John Novoa-Laurentiev
//
// For those interested in learning more about this tool, contact us at BWHMTERMS@bwh.harvard.edu .
//******************************************************************************
"""


def read_notes_and_write_features_database(input_cursor, output_cursor, input_table, output_table,
                                           input_column_name_list, output_column_name_list, symptom_no_location_file,
                                           symptom_file, location_file, symptom_no_location_category_list,
                                           symptom_category_list, location_category_list):
    sql_string = f"SELECT {', '.join(input_column_name_list)} from {input_table};"
    input_cursor.execute(sql_string)

    symptom_no_location_list, symptom2category_no_location_map = load_symptoms(symptom_no_location_file)
    symptom_list, symptom2category_map = load_symptoms(symptom_file)
    location_list, location2category_map = load_locations(location_file)
    pattern_dict = compile_patterns(symptom_list, location_list, symptom_no_location_list)

    row = input_cursor.fetchone()
    row_idx = 0
    while row:  # read, split, and find matches for each note, and write each result to output table
        print(f'processing note {row_idx}...')
        feature_dict = {loc: {sym: 0 for sym in symptom_category_list} for loc in location_category_list}
        feature_dict['no_location'] = {sym: 0 for sym in symptom_no_location_category_list}

        patient_id, enc_id, note_id, note_datreal, note_date, \
            note_create_local_date, note_last_filed_date, note_status, note_text = row

        note_sents = find_sentences(note_text)

        for sent_idx, sent_text in enumerate(note_sents):
            match_results = find_symptoms(sent_text.lower(), pattern_dict)
            for location, location_match_dict in match_results.items():
                for symptom, match_data_dict in location_match_dict.items():
                    if location == 'no_location':
                        feature_dict['no_location'][symptom2category_no_location_map[symptom]] = 1
                    else:
                        feature_dict[location2category_map[location]][symptom2category_map[symptom]] = 1

        feature_data = []
        for sym in symptom_no_location_category_list:
            feature_data.append(feature_dict['no_location'][sym])
        for sym in symptom_category_list:
            for loc in location_category_list:
                feature_data.append(feature_dict[loc][sym])
        sql_output = f"INSERT INTO {output_table} ({', '.join(output_column_name_list)}) VALUES ({', '.join(['?'] * len(feature_data))});"
        output_cursor.execute(sql_output, tuple(feature_data))
        output_cursor.commit()

        row = input_cursor.fetchone()
        row_idx += 1


def compile_patterns(symptom_list, location_list, symptom_no_location_list):
    compiled_patterns = {}

    # compile patterns
    for symptom in symptom_no_location_list:
        symptom_no_location_patterns = [
            re.compile('regex pattern 1'),
            re.compile('regex pattern 2')
        ]
        symptom_no_location_neg_patterns = [
            re.compile('regex pattern 1'),
            re.compile('regex pattern 2')
        ]

    compiled_patterns['no_location'] = {'pos': symptom_no_location_patterns,
                                        'neg': symptom_no_location_neg_patterns}

    # run through all locations
    for location in location_list:
        for symptom in symptom_list:
            symptom_location_patterns = [
                re.compile('regex pattern 1'),
                re.compile('regex pattern 2')
            ]
            symptom_location_neg_patterns = [
                re.compile('regex pattern 1'),
                re.compile('regex pattern 2')
            ]

        compiled_patterns[location] = {'pos': symptom_location_patterns,
                                       'neg': symptom_location_neg_patterns}

    return compiled_patterns


def find_symptoms(note_text, pattern_dict):
    found_terms = {}

    # run through and find matches for compiled regex patterns
    for location, pattern_dict in pattern_dict.items():
        for neg_pattern in pattern_dict['neg']:
            matches = neg_pattern.findall(note_text, overlapped=True)
            for symptom_match in matches:
                found_terms[location][symptom_match]['neg'] += 1
        for pos_pattern in pattern_dict['pos']:
            matches = pos_pattern.findall(note_text, overlapped=True)
            for symptom_match in matches:
                found_terms[location][symptom_match]['pos'] += 1

    return found_terms


def find_sentences(raw):
    # segment a note into sentences
    note_sentences = segment_note(raw)

    return note_sentences


if __name__ == '__main__':
    input_table = '[dbo].[input_table]'
    output_table = '[dbo].[output_table]'
    server = 'server'
    database = 'db'
    username = 'username'
    password = 'pw'
    symptom_no_location_file = 'lexicon/symptom_no_location_list.csv'
    symptom_file = 'lexicon/symptom_list.csv'
    location_file = 'lexicon/location_list.csv'

    input_cursor = connect_db(server, database, username, password)
    output_cursor = connect_db(server, database, username, password)
    input_column_names = ['PatientID', 'NoteID', 'etc']
    output_column_names = [
        'PatientID', 'PatientEncounterID', 'NoteID', 'symptom_Shortness_Of_Breath',
        'symptom_Calf_Numbness', 'symptom_Leg_Numbness', 'etc'
    ]

    # symptom and location categories, in order as they appear in output table columns
    symptom_no_location_category_list = ['lightheadedness', 'shortness of breath', 'etc']
    symptom_category_list = ['pain', 'numbness', 'etc']
    location_category_list = ['calf', 'leg', 'etc']
    read_notes_and_write_features_database(input_cursor, output_cursor, input_table, output_table, input_column_names,
                                           output_column_names, symptom_no_location_file, symptom_file, location_file,
                                           symptom_no_location_category_list, symptom_category_list,
                                           location_category_list)
