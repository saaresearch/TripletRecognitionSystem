# get_predict("/home/artem/pdd/index.jpeg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros.jpg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros2.jpg")
# get_predict("/home/artem/pdd/script_test/xloros/xloros3.jpg")

# print(" ")

# get_predict("/home/artem/pdd/script_test/health/health1.jpeg")
# get_predict("/home/artem/pdd/script_test/health/health2.jpg")
# get_predict("/home/artem/pdd/script_test/health/health3.jpg")

# print (" ")

# get_predict("/home/artem/pdd/script_test/gnil/gnil.jpg")
# get_predict("/home/artem/pdd/script_test/gnil/gnil1.jpg")
# get_predict("/home/artem/pdd/script_test/gnil/gnil3.jpg")

# print(" ")

# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")
# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")
# get_predict("/home/artem/pdd/script_test/apopleks/apop1.jpg")

# get_predict("/home/artem/pdd/script_test/health_corn/health_1.jpg")
# get_predict("/home/artem/pdd/script_test/health_corn/health_2.jpg")

# get_predict("/home/artem/pdd/script_test/wheat_yellow_rust/yellow_rust1.jpg")

# get_predict("/home/artem/pdd/script_test/health_weat/health.jpg")
import json
data = {
    "president": {
        "name": "Zaphod Beeblebrox",
        "species": "Betelgeusian"
    }
}

print(type(data))
with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)

 # print(distances, indices)
    # print(indices.ravel().__dir__())
    # print(indices.data)
    # print(y_pred)
    # print(len(knn._y[indices[0]]))


    # for target in knn.kneighbors(embedding[[0]], n_neighbors=10):
    # # knn._y[indices[0]],distances:
        
    #     print(target)
    #     print(target.shape)
    # print(classes_name[knn._y[indices[0]]])
# with open('classname.txt', 'w') as filehandle:
    #     for listitem in test_ds.classes:
    #      filehandle.write('%s\n' % listitem)






