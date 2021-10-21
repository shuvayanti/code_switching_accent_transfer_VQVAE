import textgrid
import os

def process_textgrid(textgrid_path):
    """
    extract phone and word alignments from textgrid file
    """
    assert textgrid_path.endswith(".TextGrid")
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    # print(tg)
    # print(tg[0])
    # print(tg[1])
    phones, words = [], []
    words_intervaltier = tg[0]
    phones_intervaltier = tg[1]
    for word in words_intervaltier:
        # print(bool(word.mark))
        words.append({
            "word": word.mark if word.mark else 'SILENCE',
            "start": word.minTime,
            "end": word.maxTime,
        })
    for phone in phones_intervaltier:
        # print(bool(phone.mark))
        phones.append({
            "phone": phone.mark,
            "start": phone.minTime,
            "end": phone.maxTime,
        })
    return phones, words
    
def main():
    file_path="../siwis_database/alignment/"
    
    
    for textgrid_file in os.listdir(file_path+"IT-textgrid/"):
        #textgrid = file
        #print(textgrid_file)
        folder = textgrid_file.split('.')[0][:-4]+'/'
        try:
            os.mkdir("../siwis_database/alignment/preprocess/IT/"+folder)
        except:
            pass
        write_file=open("../siwis_database/alignment/preprocess/IT/"+folder+textgrid_file.split('.')[0]+'.txt','w')
        #print(write_file)
        phones, words = process_textgrid(file_path+"IT-textgrid/"+textgrid_file)
        
        #print(words)
        for i,dictionary in enumerate(words):
            #print(dictionary['word']+'\t'+str(dictionary['start'])+'\t'+str(dictionary['end'])+'\n')
            
            write_file.write(dictionary['word']+'\t'+str(phones[i]['phone'])+'\t'+str(dictionary['start'])+'\t'+str(dictionary['end'])+'\n')
        
    for textgrid_file in os.listdir(file_path+"EN-textgrid/"):
        #print(textgrid_file)
        folder = textgrid_file.split('.')[0][:-4]+'/'
        try:
            os.mkdir("../siwis_database/alignment/preprocess/EN/"+folder)
        except:
            pass
        write_file=open("../siwis_database/alignment/preprocess/EN/"+folder+textgrid_file.split('.')[0]+'.txt','w')
        #print(write_file)
        phones, words = process_textgrid(file_path+"EN-textgrid/"+textgrid_file)
        
        #print(words)
        for i,dictionary in enumerate(words):
            #print(dictionary['word']+'\t'+str(dictionary['start'])+'\t'+str(dictionary['end'])+'\n')
            
            write_file.write(dictionary['word']+'\t'+str(phones[i]['phone'])+'\t'+str(dictionary['start'])+'\t'+str(dictionary['end'])+'\n')
        
if __name__ == '__main__':
    main()
