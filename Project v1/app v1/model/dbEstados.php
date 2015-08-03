<?php
/**
 * Description of buscar
 *
 * @author durso
 */
namespace app\model;
use app\model\db;


class dbEstados extends db{
    protected $table = "estados";
    
    public function select(){

        $sql = "SELECT id, nome FROM estados";
        return $this->query($sql);
    }
}