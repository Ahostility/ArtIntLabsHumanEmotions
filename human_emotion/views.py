# from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from random import randint
from django.http.response import HttpResponse
from emotionAML.modules.predict import predict
from human_emotion.model import open_result_file

# Create your views here.

@csrf_exempt
def emotionPOST(request):
    if request.method == 'GET':
        return JsonResponse({'name': 'Neural'})

    if request.method == 'POST':
        uploadedFile = request.FILES["file"]

        if uploadedFile != None:
            fs = FileSystemStorage(location='files')
            filename = fs.save(str(randint(1000, 100000)) + uploadedFile.name, uploadedFile)  # audiofile
            file_url = str('./files/') + fs.url(filename)

            resultAML = predict(file_url, 4)
            resultNLP = open_result_file(file_url)

            # return HttpResponse(resultNLP)
            return JsonResponse({
                'emotionAML': resultAML,
                'emotionNLP': resultNLP,
                'mafiaLie': None
                                 }, status= 200)

        else:
            return JsonResponse({
                'message': 'Unexpected file field'
            }, status=500)
