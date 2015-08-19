<?php

/**
 * Description of layout
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;



class body extends paired{

    public function __construct(){
        parent::__construct();
        $this->tag = "body";
    }
  
  
}
