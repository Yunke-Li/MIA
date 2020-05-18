def getEDES(f_list, file_path):
    for i in f_list:
        if 'cfg' in i:
            with open(file_path+'/'+i, 'r') as config:
                text = config.readlines()
                for t in text:
                    if 'ED' in t:
                        edSlice = str(int(t[4:]))
                        edSlice = edSlice.zfill(2)
                    if 'ES' in t:
                        esSlice = str(int(t[4:]))
                        esSlice = esSlice.zfill(2)

    return esSlice, edSlice