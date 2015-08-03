<?php

namespace app\controller;
use app\controller\base\controller;
use library\layout\components\template;
use app\model\dbMedicos, app\model\dbAvaliacoes;
use library\layout\elements\block;
use library\layout\elements\script;
use library\utils;
use app\request;
use library\layout\components\medicoBox;


class medico extends controller{

    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }

    public function index(){        
        $navbar = new template("templates/navbar","nav");
        $navbar->addClassName("navbar navbar-default navbar-fixed-top");
        try{
            $this->db = new dbMedicos();
            $opts = $this->db->buscarQuery($this->query);
            $flag = $this->db->select($opts);
            if(!$flag){
                throw new \Exception("Nao foi possivel conectar a base de dados");
            }
            $res = $this->db->fetchOne();
        } catch(\Exception $e) {
            request::redirectError($e->getMessage());
        }
        $section = new block("", array("first"), "section");
        $medico = new medicoBox($res);
        $opts = array();
        $rows = $this->queryDb($opts);
        $medico->addReview($rows);
        $section->addChild($medico);
        $footer = new template("templates/footer");
        $this->layout->addChild($navbar);
        $this->layout->addChild($section);
        $this->layout->addChild($footer);
        script::loadMore($opts,"reviewBox");
        $this->html();
    }
    public function ajax(){
        request::redirectError("Falha ao carregar o arquivo");
        $rows = $this->queryDb();
        if($this->db->numRows() > 0){
            $html = "";
            foreach($rows as $row){
                $html .= reviewRow($row);
            }
            echo $html;
        } else {
            echo "false";
        }
    }
    private function queryDb(array &$opts = array()){
        try{
            $this->db = new dbAvaliacoes();
            $opts = $this->db->avaliacoesQuery($this->query);
            $flag = $this->db->select($opts);
            if(!$flag){
                throw new \Exception("Nao foi possivel conectar a base de dados");
            }
            $rows = $this->db->fetchAll();
        } catch(\Exception $e) {
            request::redirectError($e->getMessage());
        }
        return $rows;
    }
    
    
    

}