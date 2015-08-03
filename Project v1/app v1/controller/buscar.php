<?php

namespace app\controller;
use app\controller\base\controller;
use library\layout\components\template;
use library\layout\elements\script;
use library\layout\elements\block;
use app\model\dbMedicos;
use app\model\dbEstados;
use app\request;
use library\layout\components\resultBox;
use library\layout\components\formBusca;



class buscar extends controller{

    public function __construct($action,$query) {
        parent::__construct($action,$query);
    }
    
    public function index(){
        $navbar = new template("templates/navbar","nav");
        $navbar->addClassName("navbar navbar-default navbar-fixed-top");
        
        try{
            $this->db = new dbEstados();
            $flag = $this->db->select();
            if(!$flag){
                throw new \Exception("Nao foi possivel conectar a base de dados");
            }
            $res = $this->db->fetchAll();   
        } catch(\Exception $e) {
            request::redirectError($e->getMessage());
        }
        $form = new formBusca($res);
        $form->setId(false, "formBusca");
        $section = new template("buscar/index","section");
        $section->addClassName("first");
        $section->addElement($form);
        $section->renderTemplate();
        $footer = new template("templates/footer");
        $this->layout->addChild($navbar);
        $this->layout->addChild($section);
        $this->layout->addChild($footer);
        $this->html();
    }

    public function resultado(){
        
        $navbar = new template("templates/navbar","nav");
        $navbar->addClassName("navbar navbar-default navbar-fixed-top");
        $opts = array();
        $res = $this->queryDb($opts);
        $resultado = new template("buscar/resultado","section");           
        $resultado->addClassName("first");    
        if($this->db->numRows() > 0){
            $resultBox = new resultBox($res);
        } else{
            $resultBox = new block("Sua busca nao achou nenhum resultado",array(),"div");
        }
        $resultBox->setId(false, "resultBox");
        $resultado->addElement($resultBox);
        $resultado->renderTemplate();
        $footer = new template("templates/footer");
        $this->layout->addChild($navbar);
        $this->layout->addChild($resultado);
        $this->layout->addChild($footer);
        script::loadMore($opts,"resultBox");
        $this->html();
       
    }
    public function ajax(){
        if(!request::isAjax()){
            request::redirectError("pagina indisponivel");
        }
        $res = $this->queryDb();
        if($this->db->numRows() > 0){
            $resultBox = new resultBox($res);
            echo $resultBox->getValue();
        } else {
            echo "false";
        }
        exit;

    }
    
    private function queryDb(array &$opts = array()){
        try{
            $this->db = new dbMedicos();
            $opts = $this->db->buscarQuery($this->query);
            $flag = $this->db->select($opts);
            if(!$flag){
                throw new \Exception("Nao foi possivel conectar a base de dados");
            }
            $res = $this->db->fetchAll();   
        } catch(\Exception $e) {
            request::redirectError($e->getMessage());
        }
        return $res;
    }
    
    

}