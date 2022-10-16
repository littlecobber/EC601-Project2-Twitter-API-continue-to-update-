from google.cloud import vision_v1

def sample_batch_annotate_files(
    storage_uri="gs://cloud-samples-data/vision/document_understanding/kafka.pdf",
):
    """Perform batch file annotation."""
    mime_type = "application/pdf"

    client = vision_v1.ImageAnnotatorClient()

    gcs_source = {"uri": storage_uri}
    input_config = {"gcs_source": gcs_source, "mime_type": mime_type}
    features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    # The service can process up to 5 pages per document file.
    # Here we specify the first, second, and last page of the document to be
    # processed.
    pages = [1, 2, -1]
    requests = [{"input_config": input_config, "features": features, "pages": pages}]

    response = client.batch_annotate_files(requests=requests)
    for image_response in response.responses[0].responses:
        print(u"Full text: {}".format(image_response.full_text_annotation.text))
        for page in image_response.full_text_annotation.pages:
            for block in page.blocks:
                print(u"\nBlock confidence: {}".format(block.confidence))
                for par in block.paragraphs:
                    print(u"\tParagraph confidence: {}".format(par.confidence))
                    for word in par.words:
                        print(u"\t\tWord confidence: {}".format(word.confidence))
                        for symbol in word.symbols:
                            print(
                                u"\t\t\tSymbol: {}, (confidence: {})".format(
                                    symbol.text, symbol.confidence
                                )
                            )
                            