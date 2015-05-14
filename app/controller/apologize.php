<?php

namespace app\controller;
use app\controller\base\controller as baseController;

class apologize extends baseController{

    public function __construct($action,$uri,$error = false) {
        if(is_array($error)){
            foreach($error as $msg){
                $this->errorMsg[] = $msg;
            }
        } else {
            $this->errorMsg[] = $error;

        }
        parent::__construct($action,$uri);
        
    }

    public function index(){
        $this->addView();
        $this->view->files($this->render);
        $this->view->assign("errors",$this->errorMsg);
        $this->view->render();
    }

}