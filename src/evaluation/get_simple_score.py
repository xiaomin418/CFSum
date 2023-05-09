# To get one sentence rouge score.
# [test]-hyp-text-gate-model_7_15000.txt(0.43914)
# [test]-hyp-text-global-gate-model_9_5000.txt(0.43741)
# [dev]-hyp-text-gate-model_7_15000.txt(0.45541)
# [dev]-hyp-text-global-gate-model_9_5000.txt(0.46395)
import os
import pickle
import numpy as np
# from matplotlib import pyplot as plt
import math
import sys
sys.path.append('../')

tmp_ref_dir = '../result/tmp-ref/'
tmp_hyp_dir = '../result/tmp-hyp/'
# aimed_bertokenizer = ['./images_test/1603.jpg', './images_test/1890.jpg', './images_test/1130.jpg', './images_test/439.jpg', './images_test/300.jpg', './images_test/1853.jpg', './images_test/370.jpg', './images_test/150.jpg', './images_test/631.jpg', './images_test/1862.jpg', './images_test/1082.jpg', './images_test/1029.jpg', './images_test/1521.jpg', './images_test/788.jpg', './images_test/97.jpg', './images_test/1717.jpg', './images_test/454.jpg', './images_test/1896.jpg', './images_test/1342.jpg', './images_test/1020.jpg', './images_test/110.jpg', './images_test/796.jpg', './images_test/1235.jpg', './images_test/1207.jpg', './images_test/1227.jpg', './images_test/1805.jpg', './images_test/1560.jpg', './images_test/401.jpg', './images_test/1933.jpg', './images_test/1863.jpg', './images_test/378.jpg', './images_test/1666.jpg', './images_test/272.jpg', './images_test/1633.jpg', './images_test/959.jpg', './images_test/197.jpg', './images_test/1733.jpg', './images_test/909.jpg', './images_test/72.jpg', './images_test/1665.jpg', './images_test/1884.jpg', './images_test/1794.jpg', './images_test/1596.jpg', './images_test/1332.jpg', './images_test/831.jpg', './images_test/256.jpg', './images_test/440.jpg', './images_test/414.jpg', './images_test/1482.jpg', './images_test/1482.jpg', './images_test/1504.jpg', './images_test/1317.jpg', './images_test/1263.jpg', './images_test/703.jpg', './images_test/1781.jpg', './images_test/1646.jpg', './images_test/1245.jpg', './images_test/710.jpg', './images_test/1735.jpg', './images_test/626.jpg', './images_test/687.jpg', './images_test/870.jpg', './images_test/299.jpg', './images_test/128.jpg', './images_test/1349.jpg', './images_test/1569.jpg', './images_test/1818.jpg', './images_test/1827.jpg', './images_test/924.jpg', './images_test/1768.jpg', './images_test/1956.jpg', './images_test/513.jpg', './images_test/1660.jpg', './images_test/629.jpg', './images_test/217.jpg', './images_test/1642.jpg', './images_test/1295.jpg', './images_test/886.jpg', './images_test/68.jpg', './images_test/694.jpg', './images_test/171.jpg', './images_test/1584.jpg', './images_test/992.jpg', './images_test/53.jpg', './images_test/1619.jpg', './images_test/1634.jpg', './images_test/1351.jpg', './images_test/1312.jpg', './images_test/1781.jpg', './images_test/1186.jpg', './images_test/1560.jpg', './images_test/1103.jpg', './images_test/1363.jpg', './images_test/1864.jpg', './images_test/683.jpg', './images_test/28.jpg', './images_test/621.jpg', './images_test/1294.jpg', './images_test/928.jpg', './images_test/1362.jpg', './images_test/1757.jpg', './images_test/783.jpg', './images_test/1983.jpg', './images_test/1831.jpg', './images_test/416.jpg', './images_test/1128.jpg', './images_test/657.jpg', './images_test/1107.jpg', './images_test/1075.jpg', './images_test/843.jpg', './images_test/274.jpg', './images_test/369.jpg', './images_test/681.jpg', './images_test/122.jpg', './images_test/1162.jpg', './images_test/1419.jpg', './images_test/979.jpg', './images_test/1838.jpg', './images_test/589.jpg', './images_test/45.jpg', './images_test/149.jpg', './images_test/1783.jpg', './images_test/780.jpg', './images_test/1739.jpg', './images_test/61.jpg', './images_test/919.jpg', './images_test/919.jpg', './images_test/1108.jpg', './images_test/372.jpg', './images_test/534.jpg', './images_test/998.jpg', './images_test/785.jpg', './images_test/1401.jpg', './images_test/1699.jpg', './images_test/1472.jpg', './images_test/1418.jpg', './images_test/1749.jpg', './images_test/1749.jpg', './images_test/1218.jpg', './images_test/596.jpg', './images_test/1637.jpg', './images_test/13.jpg', './images_test/199.jpg', './images_test/1804.jpg', './images_test/929.jpg', './images_test/1331.jpg', './images_test/1446.jpg', './images_test/816.jpg', './images_test/311.jpg', './images_test/1318.jpg', './images_test/284.jpg', './images_test/940.jpg', './images_test/1044.jpg', './images_test/694.jpg', './images_test/1015.jpg', './images_test/350.jpg', './images_test/908.jpg', './images_test/428.jpg', './images_test/1918.jpg', './images_test/277.jpg', './images_test/1685.jpg', './images_test/984.jpg', './images_test/1155.jpg', './images_test/482.jpg', './images_test/173.jpg', './images_test/1713.jpg', './images_test/1343.jpg', './images_test/145.jpg', './images_test/874.jpg', './images_test/1231.jpg', './images_test/1560.jpg', './images_test/995.jpg', './images_test/1479.jpg', './images_test/318.jpg', './images_test/1356.jpg', './images_test/1640.jpg', './images_test/1566.jpg', './images_test/911.jpg', './images_test/895.jpg', './images_test/1924.jpg', './images_test/1381.jpg', './images_test/1116.jpg', './images_test/597.jpg', './images_test/1276.jpg', './images_test/518.jpg', './images_test/607.jpg', './images_test/866.jpg', './images_test/590.jpg', './images_test/912.jpg', './images_test/1587.jpg', './images_test/231.jpg', './images_test/1485.jpg', './images_test/664.jpg', './images_test/39.jpg', './images_test/1178.jpg', './images_test/557.jpg', './images_test/319.jpg', './images_test/1389.jpg', './images_test/95.jpg', './images_test/1108.jpg', './images_test/1506.jpg', './images_test/777.jpg', './images_test/1002.jpg', './images_test/183.jpg', './images_test/483.jpg', './images_test/596.jpg', './images_test/1833.jpg', './images_test/1719.jpg', './images_test/525.jpg', './images_test/853.jpg', './images_test/1019.jpg', './images_test/592.jpg', './images_test/1773.jpg', './images_test/909.jpg', './images_test/297.jpg', './images_test/586.jpg', './images_test/1013.jpg', './images_test/1711.jpg', './images_test/1121.jpg', './images_test/226.jpg', './images_test/1811.jpg', './images_test/888.jpg', './images_test/149.jpg', './images_test/1358.jpg', './images_test/1883.jpg', './images_test/1622.jpg', './images_test/1783.jpg', './images_test/1365.jpg', './images_test/172.jpg', './images_test/1006.jpg', './images_test/1538.jpg', './images_test/346.jpg', './images_test/232.jpg', './images_test/1156.jpg', './images_test/1962.jpg', './images_test/1915.jpg', './images_test/315.jpg', './images_test/1928.jpg', './images_test/1928.jpg', './images_test/1051.jpg', './images_test/1749.jpg', './images_test/450.jpg', './images_test/862.jpg', './images_test/609.jpg', './images_test/991.jpg', './images_test/1842.jpg', './images_test/1063.jpg', './images_test/287.jpg', './images_test/648.jpg', './images_test/1963.jpg', './images_test/1904.jpg', './images_test/1360.jpg', './images_test/124.jpg', './images_test/1285.jpg', './images_test/262.jpg', './images_test/868.jpg', './images_test/293.jpg', './images_test/248.jpg', './images_test/1603.jpg', './images_test/987.jpg', './images_test/1528.jpg', './images_test/428.jpg', './images_test/580.jpg', './images_test/41.jpg', './images_test/486.jpg', './images_test/201.jpg', './images_test/1452.jpg', './images_test/567.jpg', './images_test/1537.jpg', './images_test/809.jpg', './images_test/571.jpg', './images_test/1188.jpg', './images_test/247.jpg', './images_test/760.jpg', './images_test/1926.jpg', './images_test/565.jpg', './images_test/582.jpg', './images_test/807.jpg', './images_test/1005.jpg', './images_test/797.jpg', './images_test/1337.jpg', './images_test/912.jpg', './images_test/273.jpg', './images_test/1822.jpg', './images_test/1470.jpg', './images_test/292.jpg', './images_test/1769.jpg', './images_test/371.jpg', './images_test/110.jpg', './images_test/1894.jpg', './images_test/1103.jpg', './images_test/353.jpg', './images_test/361.jpg', './images_test/134.jpg', './images_test/278.jpg', './images_test/1506.jpg', './images_test/174.jpg', './images_test/1908.jpg', './images_test/40.jpg', './images_test/1400.jpg', './images_test/1435.jpg', './images_test/270.jpg', './images_test/1147.jpg', './images_test/1062.jpg', './images_test/1547.jpg', './images_test/885.jpg', './images_test/1656.jpg', './images_test/1242.jpg', './images_test/1194.jpg', './images_test/121.jpg', './images_test/1411.jpg', './images_test/1866.jpg', './images_test/1013.jpg', './images_test/498.jpg']
# aimed_my_tokenizer = ['./images_test/1640.jpg', './images_test/1608.jpg', './images_test/310.jpg', './images_test/34.jpg', './images_test/1876.jpg', './images_test/265.jpg', './images_test/265.jpg', './images_test/1018.jpg', './images_test/1812.jpg', './images_test/1450.jpg', './images_test/234.jpg', './images_test/1642.jpg', './images_test/578.jpg', './images_test/489.jpg', './images_test/1921.jpg', './images_test/106.jpg', './images_test/1663.jpg', './images_test/351.jpg', './images_test/1523.jpg', './images_test/904.jpg', './images_test/867.jpg', './images_test/1904.jpg', './images_test/1635.jpg', './images_test/1384.jpg', './images_test/303.jpg', './images_test/1157.jpg', './images_test/1825.jpg', './images_test/1360.jpg', './images_test/1154.jpg', './images_test/1148.jpg', './images_test/1186.jpg', './images_test/968.jpg', './images_test/1221.jpg', './images_test/604.jpg', './images_test/14.jpg', './images_test/1491.jpg', './images_test/290.jpg', './images_test/1579.jpg', './images_test/1550.jpg', './images_test/193.jpg', './images_test/889.jpg', './images_test/725.jpg', './images_test/558.jpg', './images_test/1979.jpg', './images_test/1975.jpg', './images_test/347.jpg', './images_test/813.jpg', './images_test/740.jpg', './images_test/1735.jpg', './images_test/1060.jpg', './images_test/477.jpg', './images_test/1260.jpg', './images_test/1806.jpg', './images_test/1262.jpg', './images_test/1437.jpg', './images_test/937.jpg', './images_test/543.jpg', './images_test/589.jpg', './images_test/1885.jpg', './images_test/781.jpg', './images_test/92.jpg', './images_test/1808.jpg', './images_test/781.jpg', './images_test/1465.jpg', './images_test/1888.jpg', './images_test/404.jpg', './images_test/1650.jpg', './images_test/648.jpg', './images_test/1694.jpg', './images_test/1142.jpg', './images_test/1605.jpg', './images_test/1650.jpg', './images_test/1241.jpg', './images_test/1636.jpg', './images_test/1464.jpg', './images_test/430.jpg', './images_test/940.jpg', './images_test/271.jpg', './images_test/1284.jpg', './images_test/481.jpg', './images_test/940.jpg', './images_test/492.jpg', './images_test/749.jpg', './images_test/320.jpg', './images_test/630.jpg', './images_test/624.jpg', './images_test/1118.jpg', './images_test/1397.jpg', './images_test/1252.jpg', './images_test/1503.jpg', './images_test/1736.jpg', './images_test/1911.jpg', './images_test/1144.jpg', './images_test/1766.jpg', './images_test/803.jpg', './images_test/77.jpg', './images_test/966.jpg', './images_test/1187.jpg', './images_test/394.jpg', './images_test/1438.jpg', './images_test/1596.jpg', './images_test/1116.jpg', './images_test/1592.jpg', './images_test/1202.jpg', './images_test/1095.jpg', './images_test/1317.jpg', './images_test/364.jpg', './images_test/129.jpg', './images_test/506.jpg', './images_test/1662.jpg', './images_test/1209.jpg', './images_test/1071.jpg', './images_test/112.jpg', './images_test/1821.jpg', './images_test/1429.jpg', './images_test/484.jpg', './images_test/1478.jpg', './images_test/1224.jpg', './images_test/137.jpg', './images_test/1562.jpg', './images_test/619.jpg', './images_test/1059.jpg', './images_test/375.jpg', './images_test/1125.jpg', './images_test/611.jpg', './images_test/1851.jpg', './images_test/1084.jpg', './images_test/1179.jpg', './images_test/1413.jpg', './images_test/1162.jpg', './images_test/756.jpg', './images_test/1508.jpg', './images_test/1600.jpg', './images_test/1616.jpg', './images_test/1322.jpg', './images_test/1801.jpg', './images_test/128.jpg', './images_test/1173.jpg', './images_test/662.jpg', './images_test/1374.jpg', './images_test/1072.jpg', './images_test/1753.jpg', './images_test/1712.jpg', './images_test/1892.jpg', './images_test/1533.jpg', './images_test/1380.jpg', './images_test/93.jpg', './images_test/1946.jpg', './images_test/530.jpg', './images_test/1629.jpg', './images_test/1261.jpg', './images_test/1395.jpg', './images_test/1697.jpg', './images_test/132.jpg', './images_test/25.jpg', './images_test/897.jpg', './images_test/1370.jpg', './images_test/385.jpg', './images_test/18.jpg', './images_test/541.jpg', './images_test/1546.jpg', './images_test/403.jpg', './images_test/1002.jpg', './images_test/113.jpg', './images_test/174.jpg', './images_test/368.jpg', './images_test/179.jpg', './images_test/1062.jpg', './images_test/775.jpg', './images_test/6.jpg', './images_test/473.jpg', './images_test/1247.jpg', './images_test/1094.jpg', './images_test/1917.jpg', './images_test/290.jpg', './images_test/1198.jpg', './images_test/1712.jpg', './images_test/961.jpg', './images_test/237.jpg', './images_test/1642.jpg', './images_test/1731.jpg', './images_test/1013.jpg', './images_test/1954.jpg', './images_test/452.jpg', './images_test/1800.jpg', './images_test/466.jpg', './images_test/1125.jpg', './images_test/779.jpg', './images_test/1933.jpg', './images_test/1425.jpg', './images_test/1006.jpg', './images_test/266.jpg', './images_test/837.jpg', './images_test/924.jpg', './images_test/279.jpg', './images_test/315.jpg', './images_test/442.jpg', './images_test/1692.jpg', './images_test/230.jpg', './images_test/1945.jpg', './images_test/941.jpg', './images_test/1190.jpg', './images_test/1340.jpg', './images_test/1677.jpg', './images_test/1216.jpg', './images_test/1749.jpg', './images_test/450.jpg', './images_test/895.jpg', './images_test/982.jpg', './images_test/781.jpg', './images_test/991.jpg', './images_test/778.jpg', './images_test/487.jpg', './images_test/989.jpg', './images_test/1167.jpg', './images_test/822.jpg', './images_test/186.jpg', './images_test/1818.jpg', './images_test/1361.jpg', './images_test/1930.jpg', './images_test/1513.jpg', './images_test/606.jpg', './images_test/1633.jpg', './images_test/452.jpg', './images_test/1272.jpg', './images_test/1718.jpg', './images_test/903.jpg', './images_test/49.jpg', './images_test/1853.jpg', './images_test/1560.jpg', './images_test/394.jpg', './images_test/795.jpg', './images_test/187.jpg', './images_test/391.jpg', './images_test/743.jpg', './images_test/1978.jpg', './images_test/698.jpg', './images_test/417.jpg', './images_test/1282.jpg', './images_test/813.jpg', './images_test/343.jpg', './images_test/1303.jpg', './images_test/1957.jpg', './images_test/1021.jpg', './images_test/1196.jpg', './images_test/389.jpg', './images_test/418.jpg', './images_test/1389.jpg', './images_test/1739.jpg', './images_test/1742.jpg', './images_test/1010.jpg', './images_test/913.jpg', './images_test/1079.jpg', './images_test/337.jpg', './images_test/871.jpg', './images_test/91.jpg', './images_test/1040.jpg', './images_test/685.jpg', './images_test/853.jpg', './images_test/732.jpg', './images_test/1986.jpg', './images_test/592.jpg', './images_test/79.jpg', './images_test/1433.jpg', './images_test/1198.jpg', './images_test/1013.jpg', './images_test/1439.jpg', './images_test/466.jpg', './images_test/300.jpg']
# aimed = aimed_bertokenizer

def generate_sent_file(hyp_file, ref_file):
    hyp_lines = open(hyp_file,'r').readlines()
    ref_lines = open(ref_file,'r').readlines()
    for i, hl, rl in zip(range(len(hyp_lines)), hyp_lines, ref_lines):
        with open(tmp_hyp_dir+str(i+1)+'.txt', 'w') as f:
            f.write(hl)
            f.close()
        with open(tmp_ref_dir+str(i+1)+'.txt', 'w') as f:
            f.write(rl)
            f.close()

def get_sents_scores(prefix_path):
    scores = {}
    for i in range(2000):
        cur_hyp_path = tmp_hyp_dir + str(i+1) + '.txt'
        cur_ref_path = tmp_ref_dir + str(i+1) + '.txt'
        cmd_str = 'files2rouge ' + cur_ref_path + ' ' + cur_hyp_path
        df = os.popen(cmd_str).read()
        if len(df.split('\n'))<5:
            scores[prefix_path + str(i + 1) + '.jpg'] = [0.0, 0.0, 0.0]
            continue
        rouge1, rouge2, rougeL = df.split('\n')[5].split(' ')[3], df.split('\n')[9].split(' ')[3], \
                                 df.split('\n')[13].split(' ')[3]
        rouge1, rouge2, rougeL = float(rouge1), float(rouge2), float(rougeL)
        if i%5==0:
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
        scores[prefix_path+str(i+1)+'.jpg'] = [rouge1, rouge2, rougeL]
    return scores

def get_detail_sentence_rouge(hyp_file, ref_file, prefix_path, mode):
    generate_sent_file(hyp_file, ref_file)
    scores = get_sents_scores(prefix_path)
    scores['mode'] = mode
    import pickle
    hyp_dir = "/".join(hyp_file.split('/')[:-2]) # /data/meihuan2/PreEMMS_checkpoints/0328-textonly/hyps
    with open('{}/{}.pickle'.format(hyp_dir,hyp_file.split('/')[-1].split('.')[0]), 'wb') as f:
        pickle.dump(scores, f)
        f.close()
    return scores

def get_multimodal_better_img_ids_from_compare_file():
    import pickle
    data = pickle.load(open('compare.pickle','rb'))
    multimodal_scores = {}
    textonly_scores = {}
    for d in data:
        if d['mode']=='multimodal':
            multimodal_scores = dict(multimodal_scores, **d)
        else:
            textonly_scores = dict(textonly_scores, **d)
    multimodal_better_imgs = []
    for k,v in multimodal_scores.items():
        if k=='mode' or 'dev' in k:
            continue
        else:
            if v > textonly_scores[k]:
                multimodal_better_imgs.append(k)
    print(multimodal_better_imgs)
    print("len better: ",len(multimodal_better_imgs))


def get_mlm_better_avg_rouge2(aiming):
    import pickle
    multimodal_scores = pickle.load(open('../result/multimodal-rouge.pickle','rb'))
    textonly_scores = pickle.load(open('../result/textonly-rouge.pickle','rb'))
    multi_origin = [43.61, 22.77, 41.03]
    text_orgin = [44.00, 22.74, 41.30]
    print("******Sampled: ******")
    multimodal_scores_list1 = []
    textonly_scores_list1 = []
    multimodal_scores_list2 = []
    textonly_scores_list2 = []
    multimodal_scores_listL = []
    textonly_scores_listL = []
    for k in multimodal_scores.keys():
        if k in aiming:
            multimodal_scores_list1.append(multimodal_scores[k][0])
            textonly_scores_list1.append(textonly_scores[k][0])
            multimodal_scores_list2.append(multimodal_scores[k][1])
            textonly_scores_list2.append(textonly_scores[k][1])
            multimodal_scores_listL.append(multimodal_scores[k][2])
            textonly_scores_listL.append(textonly_scores[k][2])

    m1 = round(sum(multimodal_scores_list1) / len(multimodal_scores_list1)*100,2)
    m2 = round(sum(multimodal_scores_list2) / len(multimodal_scores_list2)*100,2)
    mL = round(sum(multimodal_scores_listL) / len(multimodal_scores_listL)*100,2)
    t1 = round(sum(textonly_scores_list1) / len(textonly_scores_list1)*100,2)
    t2 = round(sum(textonly_scores_list2) / len(textonly_scores_list2)*100,2)
    tL = round(sum(textonly_scores_listL) / len(textonly_scores_listL)*100,2)
    cur_str = str(m1) + '(' + str(round(m1-multi_origin[0],2)) + ')' + '\t' + \
              str(m2) + '(' + str(round(m2 - multi_origin[1],2)) + ')' + '\t' + \
              str(mL) + '(' + str(round(mL - multi_origin[2],2)) + ')' + '\n' + \
              str(t1) + '(' + str(round(t1 - text_orgin[0],2)) + ')' + '\t' + \
              str(t2) + '(' + str(round(t2 - text_orgin[1],2)) + ')' + '\t' + \
              str(tL) + '(' + str(round(tL - text_orgin[2],2)) + ')' + '\n'+ \
              str(t1) + '(' + str(round((m1 - multi_origin[0])-(t1 - text_orgin[0]), 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round((m2 - multi_origin[1])-(t2 - text_orgin[1]), 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round((mL - multi_origin[2])-(tL - text_orgin[2]), 2)) + ')' + '\n'
    print("len: ", len(multimodal_scores_list1))
    print(cur_str)
    print("*******Not Sampled:********")
    multimodal_scores_list1 = []
    textonly_scores_list1 = []
    multimodal_scores_list2 = []
    textonly_scores_list2 = []
    multimodal_scores_listL = []
    textonly_scores_listL = []
    for k in multimodal_scores.keys():
        if k == 'mode':
            continue
        if k not in aiming:
            multimodal_scores_list1.append(multimodal_scores[k][0])
            textonly_scores_list1.append(textonly_scores[k][0])
            multimodal_scores_list2.append(multimodal_scores[k][1])
            textonly_scores_list2.append(textonly_scores[k][1])
            multimodal_scores_listL.append(multimodal_scores[k][2])
            textonly_scores_listL.append(textonly_scores[k][2])
    m1 = round(sum(multimodal_scores_list1) / len(multimodal_scores_list1) * 100, 2)
    m2 = round(sum(multimodal_scores_list2) / len(multimodal_scores_list2) * 100, 2)
    mL = round(sum(multimodal_scores_listL) / len(multimodal_scores_listL) * 100, 2)
    t1 = round(sum(textonly_scores_list1) / len(textonly_scores_list1) * 100, 2)
    t2 = round(sum(textonly_scores_list2) / len(textonly_scores_list2) * 100, 2)
    tL = round(sum(textonly_scores_listL) / len(textonly_scores_listL) * 100, 2)
    cur_str = str(m1) + '(' + str(round(m1 - multi_origin[0], 2)) + ')' + '\t' + \
              str(m2) + '(' + str(round(m2 - multi_origin[1], 2)) + ')' + '\t' + \
              str(mL) + '(' + str(round(mL - multi_origin[2], 2)) + ')' + '\n' + \
              str(t1) + '(' + str(round(t1 - text_orgin[0], 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round(t2 - text_orgin[1], 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round(tL - text_orgin[2], 2)) + ')' + '\n'+ \
              str(t1) + '(' + str(round((m1 - multi_origin[0])-(t1 - text_orgin[0]), 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round((m2 - multi_origin[1])-(t2 - text_orgin[1]), 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round((mL - multi_origin[2])-(tL - text_orgin[2]), 2)) + ')' + '\n'
    print("len: ", len(multimodal_scores_list1))
    print(cur_str)

def get_rouge_diff(modelA_score_file, modelB_score_file):
    Ascore = pickle.load(open(modelA_score_file, 'rb'))
    Bscore = pickle.load(open(modelB_score_file, 'rb'))
    AbetterB_score  = dict()
    for k,a_s in Ascore.items():
        if k=='mode':
            continue
        if k in Bscore:
            b_s = Bscore[k]
            aMb = a_s[2] - b_s[2]
            aMb = 1/(1+math.exp((-1)*aMb))
            AbetterB_score[k.split('/')[-1]] = aMb
    return AbetterB_score
#1. get every sentence score of textonly-model
# hyp_file, ref_file = ('../result/1226-textonly.txt', '../../MMSS4.0/corpus/test_title.txt')
# get_detail_sentence_rouge(hyp_file, ref_file, './images_test/', 'textonly')

#1.2 reivise the ../result/tmp.pickle to ../result/textonly-rouge.pickle'


#2. get every sentence score of multimodal-model
# hyp_file, ref_file = ('../result/hyp.txt', '../../MMSS4.0/corpus/test_title.txt')
# get_detail_sentence_rouge(hyp_file, ref_file, './images_test/', 'multimodal')

#2.2 reivise the ../result/tmp.pickle to ../result/multimodal-rouge.pickle'


#3. get mlm better average rouge from ../result/mutlimodal-rouge.txt and ../result/textonly-rouge.txt
# print("******************MMSS Bertokenzer:**********************")
# get_mlm_better_avg_rouge2(aimed_bertokenizer)
# print("******************MMSS SpacyTokenizer:**********************")
# get_mlm_better_avg_rouge2(aimed_my_tokenizer)

# hyp_file, ref_file = ('../result/hroyp-1227-score-diff.txt', '../../MMSS4.0/corpus/test_title.txt')
# # sd_scores = get_detail_sentence_rouge(hyp_file, ref_file, './images_test/', 'score-diff')
# # hyp_file, ref_file = ('../result/hyp-1224-textonly.txt', '../../MMSS4.0/corpus/test_title.txt')
# # txt_scores = get_detail_sentence_uge(hyp_file, ref_file, './images_test/', 'textonly')
# 4. ???
# multi_score_file, txt_score_file = ('../result/hyp-1227-score-diff.pickle', '../result/hyp-1224-textonly.pickle')
# MultiBetterTxt_score = get_rouge_diff(multi_score_file, txt_score_file)
# mlm_MultiBetterTxt_score = pickle.load(open('../result/image_contri.pickle','rb'))
# xs = []
# ys = []
# xu = [0.3775406687981454, 0.4174297935376853, 0.45842951678320004, 0.45842951678320015, 0.5, 0.5415704832167999, 0.5825702064623147, 0.6224593312018546]
# yu = [[] for _ in range(len(xu))]
# for k,v in mlm_MultiBetterTxt_score.items():
#     if k in MultiBetterTxt_score:
#         xs.append(v)
#         ys.append(MultiBetterTxt_score[k])
#         ind = xu.index(v)
#         yu[ind].append(MultiBetterTxt_score[k])
# yu = [sum(l)/len(l) for l in yu]
# xs = np.array(xs)
# ys = np.array(ys)
# xu = np.array(xu)
# yu = np.array(yu)
# plt.scatter(xs, ys)
# plt.plot(xu, yu,color='red')
# plt.show()

# 5. Compare the performance between useful images and useless images
# hyp_file, ref_file = ('/data/meihuan2/GuidedMMS_checkpoints/15-base-multi/hyp_15-base-multi_epoch_5_15500.txt',
#                       '/home/meihuan2/document/MMSS4.0/corpus/test_title.txt')
# useful_pic_path = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/useful_pic_path.pickle'
# scores = get_detail_sentence_rouge(hyp_file, ref_file, '', '44.69')

# score_save = "/".join(hyp_file.split('/')[:-1])+'/score0320.pkl'
# f=open(score_save,'wb')
# pickle.dump(scores, f)
# f.close()

# useful_pic_path = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/useful_pic_path.pickle'
# useful_pic_path = "/data/meihuan2/dataset/RefGuide/analyse_useful_pic_test0.5.pickle"
# useful_pic_path = "/data/meihuan2/dataset/SmlEGuide/useless_pic_test.pickle"
# score_save = "/data/meihuan2/PreEMMS_checkpoints/0328-textonly/hyp_model_39_62000_1648476543.pickle"
if __name__=='__main__':
    hyp_file, ref_file = ('/data/meihuan2/ReAttnMMS_checkpoints/1206-c2-ot-l6l9l3/hyps/hyp_model_12_62000_1670233244.txt',
                          '/home/meihuan2/document/MMSS4.0/corpus/test_title.txt')
    get_detail_sentence_rouge(hyp_file, ref_file, './images_test/', 'multimodal')

