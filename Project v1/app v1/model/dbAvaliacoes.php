<?php
/**
 * Description of buscar
 *
 * @author durso
 */
namespace app\model;
use app\model\db;
use library\utils;

class dbAvaliacoes extends db{
    protected $table = "avaliacoes";
    
    public function select(array $args){
        $sql = "SELECT avaliacoes.id,avaliacoes.rating,avaliacoes.comentario,DATE_FORMAT(avaliacoes.data,'%d-%m-%Y') as data,usuario.nome AS nome FROM avaliacoes INNER JOIN usuario ON avaliacoes.usuario=usuario.id";
        $opts = array();        
        $where = $this->where($args,$opts);
        $sql .= $where." LIMIT 1";
        return $this->query($sql,$opts);
    }

}