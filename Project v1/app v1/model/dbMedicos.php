<?php
/**
 * Description of buscar
 *
 * @author durso
 */
namespace app\model;
use app\model\db;

class dbMedicos extends db{
    protected $table = "medicos";
    
    public function select(array $args){

        $sql = "SELECT medicos.id,medicos.nome,medicos.crm,estados.nome AS estado, IFNULL(avg(avaliacoes.rating),0) AS rating, COUNT(avaliacoes.rating) AS avaliacoes FROM medicos INNER JOIN estados ON medicos.estado=estados.id LEFT JOIN avaliacoes ON medicos.id = avaliacoes.medico";
        $opts = array();    
        $where = $this->where($args,$opts);
        $sql .= $where." GROUP BY medicos.id LIMIT 1";
        return $this->query($sql,$opts);
    }
}
