docker-compose -p ydlj -f ai-dockerfile/docker-compose-ydlj.yaml up -d

docker-compose -p sfzy -f ai-dockerfile/docker-compose-sfzy.yaml up -d

docker-compose -p htfl -f ai-dockerfile/docker-compose-htfl-standard.yaml up -d

docker-compose -p zwfc -f ai-dockerfile/docker-compose-znfc.yaml up -d

docker-compose -p jesb -f ai-dockerfile/docker-compose-jesb-lstm.yaml up -d

docker-compose -p xsfl -f ai-dockerfile/docker-compose-xsfl-xlnet.yaml up -d

docker-compose -p case4 -f ai-dockerfile/docker-compose-ayfl-lighten.yaml up -d

docker-compose -p lawasmart -f ai-dockerfile/docker-compose-fc.yaml up -d

docker-compose -p lawaindex -f ai-dockerfile/docker-compose-fc-search.yaml up -d
