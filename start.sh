config_path='./config.yaml'
host=$(cat $config_path | shyaml get-value server.host)
port=$(cat $config_path | shyaml get-value server.port)
reload=$(cat $config_path | shyaml get-value server.reload)

uvicorn_config_path='./uvicorn_config.json'

if [ $reload = 'True' ]
then
  uvicorn server.fcs:app --reload --host $host --port $port --log-config $uvicorn_config_path
else
  uvicorn server.fcs:app --host $host --port $port --log-config $uvicorn_config_path
fi