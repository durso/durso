<?php

/**
 * Description of db
 *
 * @author durso
 */
namespace app\model;
use app\model\dbconn;
use library\utils;


class db {
    protected $instance;
    protected $stmt;
    protected $table;
    protected $numRows;

    public function __construct(){
        $this->instance = dbconn::getInstance();
        if($this->instance === false){
            throw new \Exception("Nao foi possivel conectar a base de dados");
        }
    }
    
    public function query($query,array $args = array()){
        try{
            $this->stmt = $this->instance->prepare($query); 
            $this->stmt->execute($args); 
            return true;
        } catch(\PDOException $e){
            utils::log($e->getMessage());
            return false;
        }
    }
    public function fetchAll(){
        $rows = $this->stmt->fetchAll(\PDO::FETCH_ASSOC);
        $this->numRows = count($rows);
        return $rows;
    }
    public function fetchOne(){
        $row = $this->stmt->fetch(\PDO::FETCH_ASSOC);
        return $row;
    }
    public function transaction(array $query, array $args){
        $this->instance->query('START TRANSACTION');
        try{
            foreach($query as $key => $value){
                $this->instance->prepare($value);
                $this->instance->execute($args[$key]);        
            }
        } catch(\PDOException $e){
            utils::log($e->getMessage());
            $this->instance->query('ROLLBACK');
            return false;
        }
        $this->instance->query('COMMIT');
        return true;
    }
    public function insert(array $args){
        $keys = array_keys($args);
        $opts = $this->sortArgs($keys,$args);
        $sql = "INSERT INTO ".$this->table."(".implode(", ",$keys).") VALUES (".implode(", ",array_fill(0,count($opts),"?")).")";
        return $this->query($sql,$opts);
    }
    public function update(array $args){
        $opts = array();
        $sql = "UPDATE ".$this->table." SET";
        foreach($args as $key => $value){
            if($key != "id"){
                $sql .= " ".$key."=?";
                $opts[] = $value;
            }
        }
        $opts[] = $args["id"];
        $sql .= "WHERE id=?";
        return $this->query($sql,$opts);
    }
    public function delete(array $id){
        $sql .= "DELETE FROM ".$this->table." WHERE id = ?";
    }
    protected function sortArgs($keys,$args){
        $opts = array();
        foreach($keys as $key){
            $opts[] = $args[$key];
        }
        return $opts;
    }
    protected function where($args,&$opts){
        if(!count($args)){
            return "";
        }
        $i = 0;
        $where = " WHERE";
        
        foreach($args as $key => $arg){
            $operator;
            if($i > 0){
                $where .= " AND";
            }
            if($key == "last_id"){
                $operator=" > ?";
                $key = "id";
            }elseif($key == "nome"){
                $operator=" LIKE ?";
                $arg = "%".$arg."%";
            } else {
                $operator="=?";
            }
            $table = $this->table;
            $where .= " $table.$key".$operator;
            $opts[] = $arg;
            $i++;
        }
        return $where;
    }
    public function numRows(){
        return $this->numRows;
    }
            
    public function buscarQuery($params){
        $result = array();
        if(isset($params["crm"]) && !empty($params["crm"])){
            $result["crm"] = $params["crm"]; 
        }
        if(isset($params["nome"]) && !empty($params["nome"])){
            $result["nome"] = $params["nome"];
        }
        if(isset($params["estado"]) && !empty($params["estado"])){
            $result["estado"] = $params["estado"];
        }
        if(isset($params["id"]) && !empty($params["id"])){
            $result["id"] = $params["id"];
        }
        if(isset($params["last_id"]) && !empty($params["last_id"])){
            $result["last_id"] = $params["last_id"];
        }
        return $result;
    }
    public function avaliacoesQuery($params){
        $result = array();
        if(isset($params["id"]) && !empty($params["id"])){
            $result["medico"] = $params["id"];
        }
        if(isset($params["last_id"]) && !empty($params["last_id"])){
            $result["last_id"] = $params["last_id"];
        }
        return $result;
    }   
    
 }