set -euo pipefail

echo "🚀 [1/2] FedAvg 시작"
python run_federated.py --config config/fl/fedavg.yaml "$@"


echo "🚀 [2/2] FedProx 시작"
python run_federated.py --config config/fl/fedprox.yaml "$@"


echo "✅ 모든 실험 완료!"