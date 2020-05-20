import csv
import time
from collections import defaultdict

from bs4 import BeautifulSoup
from selenium import webdriver
import selenium.webdriver.chrome.service as chrome_service

import networkx as nx


class HPO:

    def __init__(self, path="../data/"):
        self.__data_path = path.strip("")

    def set_data_path(self, path):
        self.__data_path = path

    def extract_hpo(self):
        file_path = self.__data_path + "/HPO/hp.obo"
        f = open(file_path, "r", encoding="utf8")

        term_info = {}
        hpo_id = []
        hpo_parents = defaultdict(set)
        hpo_alt_id = {}

        for line in f:
            line = line.strip('\n')
            if len(line) == 0 and len(term_info) > 0:
                # save
                # print(L[0], L[1])
                hpo_id.append(term_info["id"])
                if "is_a" in term_info:
                    hpo_parents[term_info["id"]] = term_info["is_a"]
                if "alt_id" in term_info:
                    for alt_id in term_info["alt_id"]:
                        hpo_alt_id[alt_id] = term_info["id"]

                term_info.clear()
            else:
                pos = line.find(': ')
                tag = line[:pos]
                context = line[pos + 2:]
                if tag == 'id':
                    term_info["id"] = context
                if tag == 'name':
                    term_info["name"] = context
                if tag == 'is_a':
                    if "is_a" not in term_info:
                        term_info["is_a"] = set()
                    term_info["is_a"].add(context[:10])
                if tag == 'alt_id':
                    if "alt_id" not in term_info:
                        term_info["alt_id"] = set()
                    term_info["alt_id"].add(context)
        f.close()
        return hpo_id, hpo_parents, hpo_alt_id

    def get_chpo_by_id(self, id):
        def get_header():
            header = {}
            header_file = open(self.__data_path + "/HPO/header.txt", "r")
            for line in header_file:
                l = line.strip('\n').split(': ')
                header[l[0]] = l[1]
            header_file.close()
            return header

        url = 'http://www.chinahpo.org/database.html?search=' + id + '&type=0&page=1'
        print(url, end=' ')
        # chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument('--no-sandbox')
        # browser = webdriver.Chrome('chromedriver.exe', chrome_options=chrome_options)

        service = chrome_service.Service('chromedriver.exe')
        service.start()
        capabilities = {'chrome.binary': 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                        'chromeOptions': {'args': ['--headless']}}
        driver = webdriver.Remote(service.service_url, capabilities)
        driver.get(url)
        time.sleep(1)  # Let the user actually see something!
        html_text = driver.page_source
        f = open('../data/HPO/html/id_' + id[3:] + '.html', 'w', encoding='utf8')
        f.write(html_text)
        f.close()
        # print('done.')
        self.chpo_parser(id, html_text)
        driver.quit()

        """
        # browser = webdriver.Chrome('chromedriver.exe')
        browser.maximize_window()
        browser.get(url)
        # time.sleep(1)
        html_text = browser.page_source
        f = open('../data/HPO/html/id_' + id[3:] + '.html', 'w', encoding='utf8')
        f.write(html_text)
        f.close()
    
        soup = BeautifulSoup(html_text, 'html.parser')
        try:
            element = soup.find('div', {'class': 'row main'})
            element = element.find('div')
            elements = element.find_all('p')
            print(elements)
        except:
            print('none')
        """

    def chpo_parser(self, id, html_text, non_list=[]):
        soup = BeautifulSoup(html_text, 'html.parser')
        L = [id]
        try:
            element = soup.find('div', {'class': 'row main'})
            element = element.find('div')
            elements = element.find_all('p')
            # print(elements)
            for tag, item in elements:
                L.append(item.string.strip())
            # if len(L) > 0:
            #     print(L)
        except:
            print('none:', id)
            non_list.append(id)
        return L

    def extract_chpo(self):
        _, __, list_id = self.extract_hpo()
        non_list = []
        f_chpo = open(self.__data_path + '/HPO/chpo.txt', 'w', encoding='utf8')
        for id in list_id:
            file_path = self.__data_path + '/HPO/html/id_' + id[3:] + '.html'
            try:
                f = open(file_path, 'r', encoding='utf8')
                try:
                    L = self.chpo_parser(id, f.read(), non_list)
                    f_chpo.write('|'.join(L) + '\n')
                except AttributeError:
                    print('error:', id)
                finally:
                    f.close()
            except IOError:
                print('no such file:', file_path)
        print(len(non_list))
        f_chpo.close()
        return

    def get_chpo(self, left=0, right=9999999):
        _, __, list_id = self.extract_hpo()
        index = 0
        for id in list_id:
            if int(id[3:]) < left or int(id[3:]) >= right:
                continue
            self.get_chpo_by_id(id)
            index += 1

    def read_hpo_annotation(self):
        with open(self.__data_path + "/HPO/phenotype_annotation.tab") as tsv:
            """
                *       Column	Content	Required	Example
                *   0	DB	required	OMIM, ORPHA, DECIPHER, MONDO
                *   1	DB_Object_ID	required	154700
                *   2	DB_Name	required	Achondrogenesis, type IB
                *   3	Qualifier	optional	NOT
                *   4	HPO_ID	required	HP:0002487
                *   5	DB_Reference	required	OMIM:154700 or PMID:15517394
                *   6	Evidence_Code	required	IEA
                *   7	Onset modifier	optional	HP:0003577
                *   8	Frequency	optional	HP:0003577 or 12/45 or 22%
                *   9	Sex	optional	MALE or FEMALE
                *   10	Modifier	optional	HP:0025257(“;”-separated list)
                *   11	Aspect	required	P, C, or I
                *   12	Date_Created	required	YYYY-MM-DD
                *   13	Assigned_By	required	HPO
            """
            for line in csv.reader(tsv, dialect="excel-tab"):
                print(line[0], line[1], line[4], line[5], line[2])

    def build_hpo_diseases_network(self):
        hpo_by_id, _, __ = self.extract_hpo()
        G = nx.Graph()
        f = open(self.__data_path + "/graph_data/test.node", "w", encoding="utf8")
        with open(self.__data_path + "/HPO/phenotype_annotation.tab") as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):
                # print(line[0], line[1], line[4], line[5], line[2])
                hpo_id = line[4]
                disease_id = line[0] + ":" + line[1]
                G.add_node(disease_id, name=line[2], db=line[0])
                G.add_node(hpo_id, name=hpo_by_id[line[4]], db="HPO")
                G.add_edge(hpo_id, disease_id)
                f.write(hpo_id + "\t" + disease_id + "\n")
        # with open(self.__data_path + "/HPO/phenotype_annotation.tab") as tsv:
        #     for line in csv.reader(tsv, dialect="excel-tab"):
        #         # print(line[0], line[1], line[4], line[5], line[2])
        #         G.add_edge(line[4], line[5])
        f.close()
        print(G.number_of_nodes())
        print(G.number_of_edges())
        nx.write_edgelist(G, self.__data_path + "/graph_data/test.edgelist")


if __name__ == '__main__':
    # extract_hpo()
    # get_chpo()
    # extract_chpo()
    hpo = HPO("../data/")
    # hpo.read_hpo_annotation()
    hpo.build_hpo_diseases_network()
