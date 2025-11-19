## Stock Data DevOps Pipeline

### Pipeline Diagram:
![Pipeline Diagram](pipeline.png)


### Features:

* Infrastructure as Code: Terraform for AWS infrastructure
* CI/CD Pipeline: GitHub Actions with pre-merge checks and automatic API deployment
* Data Processing: ETL with AWS Glue for large files
* Model training: SageMaker to train time series model using Prophet
* API Service: FastAPI containerized on ECS with Application Load Balancer
* Security: Strict IAM roles/security groups, encrypion on S3 buckets
* Logging: Cloudwatch logging
* CI/CD: pre and post merge tests for ensuring stability and seamlessness of deployments

### Using the API:

```bash
#get URL of ALB
ALB_URL=$(terraform -chdir=terraform output -raw alb_url)
echo "$ALB_URL"
```

```bash
#health check
curl $ALB_URL/health

#model info
curl $ALB_URL/model/info

#single date prediction
curl -X POST $ALB_URL/predict \
    -H "Content-Type: application/json" \
    -d '{"date": "2024-12-31"}'

#batch predition (range)
curl -X POST $ALB_URL/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"start_date": "2024-12-01", "end_date": "2024-12-31"}'
```