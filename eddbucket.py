from skinet.utils.athena_tools import *

class eddbucket:
    RES = ''
    CID = '106be8941463b0faa93ec0aa63fc14a9f98fb9836359996ee9ab97945aecf1da'
    
    def __init__(self):
        self.RES = athena_connection(vlevel=0)
        
    def getGrant(self, target_bucket = 'sktelecom-discovery', target_db = 'gb09', target_prefix = ''):
        s3 = self.RES.s3res
        bucket = s3.Bucket(target_bucket)
        targetObj = bucket.objects.filter(Prefix = target_db + '/' + target_prefix)
        
        for obj in targetObj:
            print(obj.key)
            s3.ObjectAcl(target_bucket, obj.key).put(GrantFullControl = "id=" + self.CID)